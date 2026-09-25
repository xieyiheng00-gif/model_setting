"""Lightning indexer (DeepSeek-V3.2 DSA) and the helpers shared by DSA and CSA.

Index score:   I[t, s] = sum_h  w[t, h] * ReLU(q_I[t, h] . k_I[s])
Selection:     top-k keys per query among the causally allowed ones.
Training:      KL( head-averaged main attention || softmax(I) ) with the indexer input detached,
               over all causal keys during the dense warm-up, over the selected keys afterwards.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Rotary, masked_mean


def index_scores(q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """q: (B,nh,T,d), k: (B,S,d), w: (B,T,nh) -> (B,T,S) float32 index logits (unmasked)."""
    kT = k.to(q.dtype).transpose(-1, -2)
    out = None
    for h in range(q.shape[1]):              # loop over indexer heads: never materialise (B,nh,T,S)
        sh = F.relu(torch.matmul(q[:, h], kT))
        term = sh * w[..., h:h + 1].to(sh.dtype)
        out = term if out is None else out + term
    return out.float()


def topk_select(masked_scores: torch.Tensor, allowed: torch.Tensor, k: int) -> torch.Tensor:
    """masked_scores: (B,T,S) with -inf where not allowed. Returns bool (B,T,S): top-k ∩ allowed."""
    B, T, S = masked_scores.shape
    kk = min(k, S)
    if kk <= 0:
        return torch.zeros(B, T, S, dtype=torch.bool, device=masked_scores.device)
    idx = masked_scores.topk(kk, dim=-1).indices
    sel = torch.zeros(B, T, S, dtype=torch.bool, device=masked_scores.device).scatter_(-1, idx, True)
    return sel & allowed


def _head_avg_attention(q: torch.Tensor, k: torch.Tensor, allowed: torch.Tensor, scale: float) -> torch.Tensor:
    """Head-averaged softmax attention restricted to `allowed` (rows without keys -> 0). No grad."""
    B, H, T, _ = q.shape
    has = allowed.any(-1, keepdim=True)
    target = torch.zeros(B, T, k.shape[2], dtype=torch.float32, device=q.device)
    for h in range(H):
        s = torch.matmul(q[:, h].float(), k[:, h].float().transpose(-1, -2)) * scale
        p = torch.softmax(s.masked_fill(~allowed, float("-inf")), dim=-1)
        target += torch.where(has, p, torch.zeros_like(p))
    return target / H


def kl_to_indexer(target: torch.Tensor, index_logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """mean over valid queries of KL(target || softmax(index_logits restricted to allowed))."""
    valid = allowed.any(-1)
    safe = allowed | ~valid.unsqueeze(-1)        # rows with no allowed key: keep softmax finite
    logq = torch.log_softmax(index_logits.float().masked_fill(~safe, float("-inf")), dim=-1)
    logq = logq.masked_fill(~allowed, 0.0)
    kl = (torch.xlogy(target, target) - target * logq).sum(-1)
    v = valid.float()
    return (kl * v).sum() / v.sum().clamp(min=1.0)


def indexer_kl_loss(q: torch.Tensor, k: torch.Tensor, index_logits: torch.Tensor, allowed: torch.Tensor,
                    scale: float, max_rows: int = 0) -> torch.Tensor:
    """Indexer training loss. q,k: main attention (B,H,T,D)/(B,H,S,D), used as a detached target.
    index_logits: (B,T,S) with grad. allowed: bool broadcastable to (B,T,S)."""
    B = q.shape[0]
    allowed = allowed.expand(B, q.shape[2], k.shape[2])
    if max_rows and B > max_rows:
        q, k, index_logits, allowed = q[:max_rows], k[:max_rows], index_logits[:max_rows], allowed[:max_rows]
    with torch.no_grad():
        target = _head_avg_attention(q.detach(), k.detach(), allowed, scale)
    with torch.autocast(device_type=q.device.type, enabled=False):
        return kl_to_indexer(target, index_logits, allowed)


@torch.no_grad()
def recall_stats(q: torch.Tensor, k: torch.Tensor, allowed: torch.Tensor, sel: torch.Tensor,
                 index_logits: torch.Tensor, scale: float, k_top: int) -> dict:
    """Probe-batch diagnostics of an indexer.

    attention-mass recall: dense (all causal keys) attention probability that falls inside the
    indexer's selection, per head, averaged over queries that are actually sparse (#keys > k).
    effective sparsity: fraction of queries whose number of causal keys exceeds k.
    density: mean |selected| / |causal keys|.
    indexer_kl_dense: KL(head-avg dense attention || indexer) over all causal keys.
    """
    B, H, T, _ = q.shape
    S = k.shape[2]
    allowed = allowed.expand(B, T, S)
    sel = sel.expand(B, T, S)
    n_allowed = allowed.sum(-1)
    has = n_allowed > 0
    sparse_rows = n_allowed > k_top
    sel_f = sel.float()
    per_head_sparse, per_head_all = [], []
    avg = torch.zeros(B, T, S, dtype=torch.float32, device=q.device)
    for h in range(H):
        s = torch.matmul(q[:, h].float(), k[:, h].float().transpose(-1, -2)) * scale
        p = torch.softmax(s.masked_fill(~allowed, float("-inf")), dim=-1)
        p = torch.where(has.unsqueeze(-1), p, torch.zeros_like(p))
        mass = (p * sel_f).sum(-1)
        per_head_all.append(masked_mean(mass, has).item())
        per_head_sparse.append(masked_mean(mass, sparse_rows).item() if bool(sparse_rows.any()) else 1.0)
        avg += p
    avg /= H
    avg_mass = (avg * sel_f).sum(-1)
    return {
        "recall_per_head": per_head_sparse,
        "recall_mean": sum(per_head_sparse) / H,
        "recall_min_head": min(per_head_sparse),
        "recall_headavg_dist": masked_mean(avg_mass, sparse_rows).item() if bool(sparse_rows.any()) else 1.0,
        "recall_all_queries": sum(per_head_all) / H,
        "effective_sparsity": masked_mean(sparse_rows.float(), has).item(),
        "density": masked_mean(sel.sum(-1).float() / n_allowed.clamp(min=1).float(), has).item(),
        "indexer_kl_dense": kl_to_indexer(avg, index_logits, allowed).item(),
    }


class LightningIndexer(nn.Module):
    """Token-level lightning indexer used by DSA layers (queries and keys from the layer input)."""

    def __init__(self, d_model: int, n_heads: int, dim: int, rope_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        self.n_heads, self.dim = n_heads, dim
        self.wq = nn.Linear(d_model, n_heads * dim, bias=False)
        self.wk = nn.Linear(d_model, dim, bias=False)
        self.k_norm = nn.LayerNorm(dim)
        self.ww = nn.Linear(d_model, n_heads, bias=False)
        self.rope = Rotary(rope_dim, max_seq_len, theta) if rope_dim > 0 else None
        self.w_scale = n_heads ** -0.5 * dim ** -0.5

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.dim).transpose(1, 2)
        k = self.k_norm(self.wk(x))
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        w = self.ww(x).float() * self.w_scale
        return index_scores(q, k, w)


def softmax_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)
