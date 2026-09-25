"""C: Compressed Sparse Attention (DeepSeek-V4), adapted to 12 query heads x 64.

1. KV compressor: two streams (a, b) of entries C and gate logits Z from the layer input. Entry i
   pools 2m tokens with a per-channel softmax over [Z^a_{block i} + B^a ; Z^b_{block i-1} + B^b]:
       C_i = sum_j S^a_j * C^a_j  +  sum_j S^b_j * C^b_j           (overlapping windows, stride m)
2. Lightning indexer over compressed indexer keys (same compressor, dim c_I) picks the top-k
   compressed entries per query among the COMPLETED blocks (block i usable at t >= m(i+1)-1).
3. Core attention: one softmax over [selected compressed entries ; n_win-token sliding window of
   uncompressed a-stream entries ; learnable per-head sink logit]. Shared KV (K = V, MQA/GQA style)
   with RMSNorm on q and on KV entries; partial RoPE on the last `rope_dim` channels of q and KV, and
   inverse RoPE (position -t) on the output so values carry relative position.
Adaptations for this study (V4 uses much wider heads): kv_heads=6 so the compressor matches F's K/V
parameter count, no grouped output projection (12x64 -> 768 is already small), indexer trained with
the V3.2 KL recipe (dense warm-up over all completed blocks, then KL on the selected set).
Packed documents: a compressed entry belongs to the document of its LAST token; tokens of any other
document are masked out of its pooling softmax, and it is usable only by queries of that document.
A document that starts on the block grid therefore gets exactly the entries it would get alone. The
window stays inside the document; RoPE positions restart per document (an entry takes the position of
its last token).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .common import Ctx, RMSNorm, Rotary, head_softmax_stats
from .indexer import index_scores, indexer_kl_loss, recall_stats, softmax_scale, topk_select


class GatedCompressor(nn.Module):
    """Every m tokens -> one entry; each entry softmax-pools its own block (stream a) and the previous
    block (stream b) per channel, with learnable positional biases."""

    def __init__(self, d_in: int, d_out: int, m: int):
        super().__init__()
        self.m, self.d_out = m, d_out
        self.proj = nn.Linear(d_in, 4 * d_out, bias=False)       # [C^a | C^b | Z^a | Z^b]
        self.pos_bias_a = nn.Parameter(torch.zeros(m, d_out))
        self.pos_bias_b = nn.Parameter(torch.zeros(m, d_out))
        self.last_pool_entropy: float | None = None

    def forward(self, h: torch.Tensor, diag: bool = False, doc_id: torch.Tensor | None = None):
        """h (B,T,d) -> (compressed (B,N,d_out) fp32 with N = T // m, token-level C^a (B,T,d_out)).
        doc_id (B,T): pool only tokens of the document that owns the entry (its last token)."""
        B, T, _ = h.shape
        m, D = self.m, self.d_out
        N = T // m
        ca, cb, za, zb = self.proj(h).chunk(4, dim=-1)
        if N == 0:
            return ca.new_zeros(B, 0, D, dtype=torch.float32), ca
        L = N * m
        with torch.autocast(device_type=h.device.type, enabled=False):
            blk = lambda t: t[:, :L].float().reshape(B, N, m, D)
            ca_b, cb_b = blk(ca), blk(cb)
            za_b = blk(za) + self.pos_bias_a.float()
            zb_b = blk(zb) + self.pos_bias_b.float()
            cb_prev = F.pad(cb_b, (0, 0, 0, 0, 1, 0))[:, :N]                     # block i-1 (zeros for i=0)
            zb_prev = F.pad(zb_b, (0, 0, 0, 0, 1, 0), value=float("-inf"))[:, :N]
            if doc_id is not None:
                da = doc_id[:, :L].reshape(B, N, m)
                owner = da[:, :, -1:]                                              # doc of the entry's last token
                db = F.pad(da, (0, 0, 1, 0), value=-1)[:, :N]                      # doc ids of the previous block
                za_b = za_b.masked_fill((da != owner).unsqueeze(-1), float("-inf"))
                zb_prev = zb_prev.masked_fill((db != owner).unsqueeze(-1), float("-inf"))
            w = torch.softmax(torch.cat([za_b, zb_prev], dim=2), dim=2)         # (B,N,2m,D)
            comp = (w * torch.cat([ca_b, cb_prev], dim=2)).sum(2)
            if diag:
                ent = -(torch.xlogy(w, w)).sum(2)                                # per entry & channel
                self.last_pool_entropy = (ent / math.log(2 * m)).mean().item()
        return comp, ca


class CSAAttention(nn.Module):
    layer_code = "C"

    def __init__(self, cfg: ModelConfig, use_rope: bool = True):
        super().__init__()
        c = cfg.csa
        self.H, self.hd, self.kvh = cfg.n_heads, cfg.head_dim, c.kv_heads
        self.m, self.top_k, self.window = c.compress_ratio, c.top_k, c.window
        self.itrain = cfg.indexer
        d = cfg.d_model
        self.q_proj = nn.Linear(d, self.H * self.hd, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.norm_eps)
        self.kv_comp = GatedCompressor(d, self.kvh * self.hd, self.m)
        self.kv_norm = RMSNorm(self.hd, cfg.norm_eps)
        self.o_proj = nn.Linear(self.H * self.hd, d, bias=False)
        self.o_proj._is_residual_proj = True
        self.sink = nn.Parameter(torch.zeros(self.H)) if c.attn_sink else None
        self.rope = Rotary(c.rope_dim, cfg.max_seq_len, cfg.rope_theta) if (use_rope and c.rope_dim > 0) else None
        # indexer
        self.nI, self.cI = c.index_heads, c.index_dim
        self.idx_wq = nn.Linear(d, self.nI * self.cI, bias=False)
        self.idx_ww = nn.Linear(d, self.nI, bias=False)
        self.idx_kcomp = GatedCompressor(d, self.cI, self.m)
        self.idx_knorm = nn.LayerNorm(self.cI)
        self.idx_rope = Rotary(c.index_rope_dim, cfg.max_seq_len, cfg.rope_theta) if c.index_rope_dim > 0 else None
        self.idx_scale = self.nI ** -0.5 * self.cI ** -0.5
        self.scale = softmax_scale(self.hd)
        self.c_pad = ((self.hd + 1 + 7) // 8) * 8          # q/k width with the extra sink channel
        self.last_diag: dict | None = None

    # ---- pieces -------------------------------------------------------------------------
    def _index_logits(self, x, pos_t, pos_c, doc_id=None):
        B, T, _ = x.shape
        q = self.idx_wq(x).view(B, T, self.nI, self.cI).transpose(1, 2)
        kc, _ = self.idx_kcomp(x, doc_id=doc_id)
        kc = self.idx_knorm(kc)
        if self.idx_rope is not None:
            q, kc = self.idx_rope(q, pos_t), self.idx_rope(kc, pos_c)
        w = self.idx_ww(x).float() * self.idx_scale
        return index_scores(q, kc, w)                                           # (B,T,N)

    def _core(self, q, kc, kw, comp_mask, win):
        """Single softmax over [compressed | window tokens | sink]. q (B,H,T,c); kc (B,H,N,c); kw (B,H,T,c)."""
        B, H, T, c = q.shape
        keys = torch.cat([kc, kw], dim=2)                                       # K = V (shared entries)
        mask = torch.cat([comp_mask, win.expand(B, T, T)], dim=-1)
        if self.sink is None:
            qa, ka, va = q, keys, keys
        else:
            # sink as an extra key whose logit is sink_h for every query: append a constant-1 channel
            # to q and put sink_h/scale in that channel of the sink key (value = 0).
            cp = self.c_pad
            qa = torch.cat([q, q.new_ones(B, H, T, 1), q.new_zeros(B, H, T, cp - c - 1)], dim=-1)
            kpad = F.pad(keys, (0, cp - c))
            sink_col = (self.sink.float() / self.scale).to(q.dtype).view(1, H, 1, 1).expand(B, H, 1, 1)
            sink_k = torch.cat([q.new_zeros(B, H, 1, c), sink_col, q.new_zeros(B, H, 1, cp - c - 1)], dim=-1)
            ka = torch.cat([kpad, sink_k], dim=2)
            va = torch.cat([kpad, q.new_zeros(B, H, 1, cp)], dim=2)
            mask = torch.cat([mask, mask.new_ones(B, T, 1)], dim=-1)
        S = ka.shape[2]
        extra = (-S) % 8                                                        # keep kernels aligned
        if extra:
            ka = F.pad(ka, (0, 0, 0, extra))
            va = F.pad(va, (0, 0, 0, extra))
            mask = torch.cat([mask, mask.new_zeros(B, T, extra)], dim=-1)
        o = F.scaled_dot_product_attention(qa, ka, va, attn_mask=mask.unsqueeze(1), scale=self.scale)
        return o[..., :c]

    # ---- forward ------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, ctx: Ctx):
        B, T, _ = x.shape
        H, kvh, c, m = self.H, self.kvh, self.hd, self.m
        flags = ctx.flags
        dev = x.device
        N = T // m
        t_idx = torch.arange(T, device=dev)
        blk_end = torch.arange(N, device=dev) * m + (m - 1)                  # last token pooled by entry i
        doc_id = None if ctx.doc is None else ctx.doc.doc_id
        if ctx.doc is None:
            pos_t, pos_c = t_idx, blk_end
            comp_allowed = (blk_end.unsqueeze(0) <= t_idx.unsqueeze(1)).unsqueeze(0)          # (1,T,N)
            rel = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)
            win = ((rel >= 0) & (rel < self.window)).unsqueeze(0)                              # (1,T,T)
        else:
            doc = ctx.doc
            pos_t = doc.pos                                                                     # (B,T)
            pos_c = doc.pos[:, blk_end] if N > 0 else doc.pos[:, :0]                          # (B,N)
            # entry i is usable by query t iff complete (its last token <= t) and owned by t's document
            comp_allowed = ((blk_end.view(1, 1, N) <= t_idx.view(1, T, 1))
                            & (doc.doc_id[:, blk_end].unsqueeze(1) == doc.doc_id.unsqueeze(2)))   # (B,T,N)
            rel = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)
            win = (((rel >= 0) & (rel < self.window)).unsqueeze(0)
                   & (doc.doc_id.unsqueeze(2) == doc.doc_id.unsqueeze(1)))                     # (B,T,T)
        q = self.q_norm(self.q_proj(x).view(B, T, H, c).transpose(1, 2))        # (B,H,T,c)
        comp, ca = self.kv_comp(x, diag=flags.diag, doc_id=doc_id)
        kc = self.kv_norm(comp.view(B, N, kvh, c).transpose(1, 2))              # (B,kvh,N,c)
        kw = self.kv_norm(ca.view(B, T, kvh, c).transpose(1, 2))                # (B,kvh,T,c)
        if self.rope is not None:
            q = self.rope(q, pos_t)
            kc = self.rope(kc, pos_c) if N > 0 else kc
            kw = self.rope(kw, pos_t)
        kc, kw = kc.to(q.dtype), kw.to(q.dtype)
        rep = H // kvh
        kc, kw = kc.repeat_interleave(rep, dim=1), kw.repeat_interleave(rep, dim=1)

        xi = x.detach() if self.itrain.detach_input else x
        if N > 0:
            scores = self._index_logits(xi, pos_t, pos_c, doc_id).masked_fill(~comp_allowed, float("-inf"))
            sel = topk_select(scores, comp_allowed, self.top_k)
        else:
            scores = torch.zeros(B, T, 0, device=dev)
            sel = torch.zeros(B, T, 0, dtype=torch.bool, device=dev)
        sparse = flags.sparse and self.top_k < N
        comp_mask = sel if sparse else comp_allowed.expand(B, T, N)

        o = self._core(q, kc, kw, comp_mask, win)
        if self.rope is not None:
            o = self.rope(o, pos_t, inverse=True)
        aux = {}
        if flags.indexer_loss and torch.is_grad_enabled() and N > 0:
            aux["indexer_kl"] = indexer_kl_loss(q, kc, scores, comp_mask, self.scale, self.itrain.kl_max_rows)
        if flags.diag:
            self.last_diag = self._diag(q, kc, kw, comp_mask, win, comp_allowed, sel, scores, sparse, N)
        return self.o_proj(o.transpose(1, 2).reshape(B, T, H * c)), aux

    @torch.no_grad()
    def _diag(self, q, kc, kw, comp_mask, win, comp_allowed, sel, scores, sparse, N) -> dict:
        B, H, T, c = q.shape
        keys = torch.cat([kc, kw], dim=2)
        mask = torch.cat([comp_mask, win.expand(B, T, T)], dim=-1)
        st = head_softmax_stats(q, keys, mask, self.scale, sink=self.sink)
        # probability split between compressed entries / window tokens / sink, head-averaged
        comp_mass, win_mass = [], []
        for h in range(H):
            s = torch.matmul(q[:, h].float(), keys[:, h].float().transpose(-1, -2)) * self.scale
            s = s.masked_fill(~mask, float("-inf"))
            if self.sink is not None:
                s = torch.cat([s, s.new_full((B, T, 1), float(self.sink[h]))], dim=-1)
            p = torch.softmax(s, dim=-1)
            comp_mass.append(p[..., :N].sum(-1).mean().item())
            win_mass.append(p[..., N:N + T].sum(-1).mean().item())
        st["compressed_mass"] = sum(comp_mass) / H
        st["window_mass"] = sum(win_mass) / H
        if self.sink is not None:
            st["sink_logit"] = self.sink.float().tolist()
        if N > 0:
            st.update(recall_stats(q, kc, comp_allowed, sel, scores, self.scale, self.top_k))
        st["kv_pool_entropy"] = self.kv_comp.last_pool_entropy
        st["sparse_phase"] = bool(sparse)
        st["top_k"] = self.top_k
        return st
