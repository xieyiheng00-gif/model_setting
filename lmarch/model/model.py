"""The language model: tied embedding -> 12 x [pre-norm mixer + residual, pre-norm SwiGLU + residual]
-> final RMSNorm -> tied LM head. The mixer of each layer is chosen by the layer pattern.

Loss: the LM head + cross-entropy run in chunks of `loss_chunk_tokens` under activation checkpointing,
so the (tokens x 128,256) logits are never materialised. forward() returns SUMS (loss_sum, n_valid)
so the trainer can average over the global batch exactly as TRAINING_DATA.md prescribes."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import INDEXER_TYPES, LAYER_TYPE_NAMES, ModelConfig
from .attention import DSAAttention, SoftmaxAttention, flash_varlen_fn
from .common import Ctx, DocInfo, RMSNorm, RunFlags, SwiGLU
from .csa import CSAAttention
from .kda import KDAMixer, select_kda_backend

MIXERS = {"F": SoftmaxAttention, "S": DSAAttention, "K": KDAMixer, "C": CSAAttention}


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, idx: int, code: str, use_rope: bool):
        super().__init__()
        self.idx, self.code, self.type_name = idx, code, LAYER_TYPE_NAMES[code]
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = MIXERS[code](cfg, use_rope=use_rope)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_dim)
        self.last_diag: dict | None = None

    def forward(self, x: torch.Tensor, ctx: Ctx):
        h, aux = self.mixer(self.norm1(x), ctx)
        x = x + h
        f = self.ffn(self.norm2(x))
        x = x + f
        if ctx.flags.diag:
            with torch.no_grad():
                rms = lambda t: t.float().pow(2).mean().sqrt().item()
                self.last_diag = {
                    "act_rms": rms(x),                         # block output (residual stream)
                    "act_absmax": x.float().abs().max().item(),
                    "mixer_out_rms": rms(h),
                    "ffn_out_rms": rms(f),
                }
        return x, aux


class LM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.pattern = cfg.pattern()
        hybrid = len(set(self.pattern)) > 1
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        blocks = []
        for i, code in enumerate(self.pattern):
            use_rope = not (hybrid and cfg.hybrid_softmax_nope and code in ("F", "S"))
            blocks.append(Block(cfg, i, code, use_rope))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.indexer_layers = [i for i, c in enumerate(self.pattern) if c in INDEXER_TYPES]
        self.grad_checkpointing = False
        self.z_loss_coef = 0.0
        self.loss_chunk_tokens = 2048
        self.reset_parameters()

    # ---- init ---------------------------------------------------------------------------
    def reset_parameters(self) -> None:
        """N(0, 0.02) for every Linear/Embedding; residual output projections (attention/KDA/CSA o_proj,
        SwiGLU w_down) use 0.02/sqrt(2*n_layers). Mixer-specific parameters (KDA A_log/dt_bias, CSA
        positional biases/sink, LayerNorms, convs) keep the init defined in their modules."""
        std = self.cfg.init_std
        resid = std / math.sqrt(2 * self.cfg.n_layers)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, 0.0, resid if getattr(mod, "_is_residual_proj", False) else std)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, 0.0, std)
        for mod in self.modules():
            if hasattr(mod, "reset_special_parameters"):
                mod.reset_special_parameters(self.cfg)

    def configure_runtime(self, device: torch.device, doc_mask: bool, bf16: bool) -> dict:
        """Pick kernels for this device: KDA backend (fla varlen / torch) and document-masked full attention
        (flash-attn varlen / SDPA with mask). Returns info for the run metadata."""
        info = {}
        kda = [b.mixer for b in self.blocks if b.code == "K"]
        if kda:
            backend, reason = select_kda_backend(self.cfg.kda.backend, device, varlen=doc_mask)
            for m in kda:
                m.set_backend(backend)
            info["kda_backend"], info["kda_backend_reason"] = backend, reason
        attn = [b.mixer for b in self.blocks if b.code in ("F", "S")]
        if attn:
            want = self.cfg.attn_backend
            ok = device.type == "cuda" and bf16 and flash_varlen_fn() is not None
            if want == "flash_varlen" and not ok:
                raise RuntimeError("attn_backend=flash_varlen needs CUDA, bf16 and the flash-attn package")
            use = doc_mask and ok and want in ("auto", "flash_varlen")
            for m in attn:
                m.use_flash_varlen = use
            info["attn_backend"] = "flash_varlen" if use else ("sdpa_mask" if doc_mask else "sdpa_causal")
        return info

    # ---- forward ------------------------------------------------------------------------
    def forward(self, idx: torch.Tensor, labels: torch.Tensor | None = None, flags: RunFlags | None = None,
                pos: torch.Tensor | None = None, per_token: bool = False, return_logits: bool = False,
                return_hidden: bool = False) -> dict:
        """idx (B,T); labels (B,T) next-token targets with -100 = ignore (already shifted);
        pos (B,T) positions restarting at 0 in every document segment (None = one causal sequence).
        Returns loss_sum / n_valid / row_loss / row_valid (+ tok_loss, z_sum, logits, hidden, aux_loss,
        indexer_kl)."""
        ctx = Ctx(flags or RunFlags(), DocInfo(pos) if pos is not None else None)
        x = self.embed(idx)
        kls = []
        for blk in self.blocks:
            if self.grad_checkpointing and self.training and torch.is_grad_enabled() and not ctx.flags.diag:
                x, aux = checkpoint(blk, x, ctx, use_reentrant=False)
            else:
                x, aux = blk(x, ctx)
            if "indexer_kl" in aux:
                kls.append(aux["indexer_kl"])
        x = self.norm_f(x)
        out: dict = {}
        if kls:
            kl = torch.stack([k.float() for k in kls])
            out["indexer_kl"] = kl.detach()
            out["aux_loss"] = self.cfg.indexer.loss_coef * kl.sum()
        if labels is not None:
            want_z = self.z_loss_coef > 0 and self.training
            tok, lse = chunked_ce(x, self.lm_head.weight, labels, self.loss_chunk_tokens, want_z)
            valid = labels != -100
            out["loss_sum"] = tok.sum()
            out["n_valid"] = valid.sum()
            out["row_loss"] = tok.sum(1).detach()
            out["row_valid"] = valid.sum(1)
            if per_token:
                out["tok_loss"] = tok.detach()
            if want_z:
                out["z_sum"] = (lse.pow(2) * valid).sum()
        if return_logits:
            out["logits"] = self.lm_head(x)
        if return_hidden:
            out["hidden"] = x
        return out

    # ---- bookkeeping --------------------------------------------------------------------
    def param_groups_by_layer(self) -> list[tuple[str, torch.nn.Parameter, str, str]]:
        """(name, param, layer_key, group) for every unique parameter. layer_key: 'L03' / 'embed' /
        'final_norm'; group: mixer | indexer | ffn | norm | embed."""
        out = []
        seen = set()
        for name, p in self.named_parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            if name.startswith("blocks."):
                i = int(name.split(".")[1])
                key = f"L{i:02d}"
                if ".mixer." in name:
                    grp = "indexer" if (".indexer." in name or ".idx_" in name) else "mixer"
                elif ".ffn." in name:
                    grp = "ffn"
                else:
                    grp = "norm"
            elif name.startswith("embed") or name.startswith("lm_head"):
                key, grp = "embed", "embed"
            else:
                key, grp = "final_norm", "norm"
            out.append((name, p, key, grp))
        return out

    def param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel()
        by_type: dict = {}
        for b in self.blocks:
            t = by_type.setdefault(b.type_name, {"layers": 0, "mixer": 0, "indexer": 0, "ffn": 0, "norm": 0})
            t["layers"] += 1
            for n, p in b.named_parameters():
                if n.startswith("mixer."):
                    t["indexer" if (".indexer." in n or ".idx_" in n) else "mixer"] += p.numel()
                elif n.startswith("ffn."):
                    t["ffn"] += p.numel()
                else:
                    t["norm"] += p.numel()
        return {"total": total, "embedding": emb, "non_embedding": total - emb,
                "tied_embeddings": self.cfg.tie_embeddings, "by_type": by_type, "pattern": self.pattern}


def build_model(cfg: ModelConfig) -> LM:
    return LM(cfg)


def _ce_chunk(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor, want_lse: bool):
    logits = F.linear(h, w).float()
    ce = F.cross_entropy(logits, y, ignore_index=-100, reduction="none")
    if want_lse:
        return ce, torch.logsumexp(logits, dim=-1)
    return ce, ce.new_zeros(())


def chunked_ce(h: torch.Tensor, w: torch.Tensor, y: torch.Tensor, chunk: int, want_lse: bool = False):
    """Per-token CE of the tied LM head, `chunk` tokens at a time. With grad enabled every chunk is
    checkpointed, so at most one chunk of fp32 logits exists at a time (forward and backward).
    Returns (tok_loss (B,T) fp32 with 0 at ignored positions, lse (B,T) or None)."""
    B, T, d = h.shape
    hf, yf = h.reshape(-1, d), y.reshape(-1)
    tl, ls = [], []
    for s in range(0, hf.shape[0], chunk):
        args = (hf[s:s + chunk], w, yf[s:s + chunk], want_lse)
        if torch.is_grad_enabled():
            ce, lse = checkpoint(_ce_chunk, *args, use_reentrant=False)
        else:
            ce, lse = _ce_chunk(*args)
        tl.append(ce)
        ls.append(lse)
    tok = torch.cat(tl).view(B, T)
    return tok, (torch.cat(ls).view(B, T) if want_lse else None)
