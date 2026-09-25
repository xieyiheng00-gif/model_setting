"""F: full causal softmax attention.   S: DeepSeek Sparse Attention (DSA, DeepSeek-V3.2).

Both use MHA with 12 heads x 64, RoPE (positions restart in every document segment), no biases, and
never attend across document segments of a packed row. DSA adds a lightning indexer and, after the
dense warm-up, restricts every query to its top-k keys (shared across heads, as in V3.2).

Kernels: without document info -> SDPA causal (flash); with document info -> flash-attn varlen when
available (H100, `attn_backend: auto|flash_varlen`), otherwise SDPA with a boolean document-causal
mask (memory-efficient kernel; RTX 3060 / Windows). The DSA sparse phase is computed exactly with a
boolean top-k mask on a dense kernel: quality is faithful, wall-clock is NOT that of a sparse kernel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .common import Ctx, RMSNorm, Rotary, head_softmax_stats
from .indexer import LightningIndexer, indexer_kl_loss, recall_stats, softmax_scale, topk_select


def flash_varlen_fn():
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore
        return flash_attn_varlen_func
    except Exception:
        return None


class SoftmaxAttention(nn.Module):
    layer_code = "F"

    def __init__(self, cfg: ModelConfig, use_rope: bool = True):
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        inner = cfg.n_heads * cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * inner, bias=False)
        self.o_proj = nn.Linear(inner, cfg.d_model, bias=False)
        self.o_proj._is_residual_proj = True
        self.rope = Rotary(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta) if use_rope else None
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.scale = softmax_scale(cfg.head_dim)
        self.use_flash_varlen = False          # set by LM.configure_runtime
        self.last_diag: dict | None = None

    def _qkv(self, x: torch.Tensor, ctx: Ctx):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, ctx.pos), self.rope(k, ctx.pos)
        return q, k, v

    def _out(self, o: torch.Tensor) -> torch.Tensor:
        B, H, T, D = o.shape
        return self.o_proj(o.transpose(1, 2).reshape(B, T, H * D))

    def _dense(self, q, k, v, ctx: Ctx) -> torch.Tensor:
        """Document-causal dense attention. q,k,v: (B,H,T,D) -> (B,H,T,D)."""
        if ctx.doc is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        B, H, T, D = q.shape
        if self.use_flash_varlen and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16):
            cu, max_len = ctx.doc.cu_seqlens()
            flat = lambda t: t.transpose(1, 2).reshape(B * T, H, D)
            o = flash_varlen_fn()(flat(q), flat(k), flat(v), cu, cu, max_len, max_len, causal=True)
            return o.view(B, T, H, D).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=ctx.attn_mask(T, q.device).unsqueeze(1))

    def forward(self, x: torch.Tensor, ctx: Ctx):
        q, k, v = self._qkv(x, ctx)
        if ctx.flags.diag:
            self.last_diag = head_softmax_stats(q, k, ctx.attn_mask(x.shape[1], x.device), self.scale)
        return self._out(self._dense(q, k, v, ctx)), {}


class DSAAttention(SoftmaxAttention):
    layer_code = "S"

    def __init__(self, cfg: ModelConfig, use_rope: bool = True):
        super().__init__(cfg, use_rope)
        d = cfg.dsa
        self.top_k = d.top_k
        self.itrain = cfg.indexer
        self.indexer = LightningIndexer(cfg.d_model, d.index_heads, d.index_dim, d.index_rope_dim,
                                        cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x: torch.Tensor, ctx: Ctx):
        B, T, _ = x.shape
        flags = ctx.flags
        q, k, v = self._qkv(x, ctx)
        allowed = ctx.attn_mask(T, x.device)                                  # (B|1,T,T) doc-causal
        xi = x.detach() if self.itrain.detach_input else x
        scores = self.indexer(xi, ctx.pos).masked_fill(~allowed, float("-inf"))   # (B,T,T) fp32
        sel = topk_select(scores, allowed, self.top_k)
        sparse = flags.sparse and self.top_k < T
        if sparse:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=sel.unsqueeze(1))
        else:
            o = self._dense(q, k, v, ctx)
        used = sel if sparse else allowed
        aux = {}
        if flags.indexer_loss and torch.is_grad_enabled():
            aux["indexer_kl"] = indexer_kl_loss(q, k, scores, used, self.scale, self.itrain.kl_max_rows)
        if flags.diag:
            st = head_softmax_stats(q, k, used, self.scale)          # the attention actually used
            st.update(recall_stats(q, k, allowed, sel, scores, self.scale, self.top_k))
            st["sparse_phase"] = bool(sparse)
            st["top_k"] = self.top_k
            self.last_diag = st
        return self._out(o), aux
