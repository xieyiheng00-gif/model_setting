"""Shared building blocks: runtime flags, document context, RMSNorm, SwiGLU, RoPE and statistics helpers."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RunFlags:
    """Runtime switches passed down to every mixer on each forward."""
    sparse: bool = True          # DSA/CSA: top-k sparse attention (False = dense warm-up phase)
    indexer_loss: bool = False   # compute the indexer KL loss (only when grad is enabled)
    diag: bool = False           # stash diagnostic statistics on the modules (probe batch only)


class DocInfo:
    """Document segments of a packed batch, derived from per-token positions (restarting at 0 in every
    segment). Masks are built once per forward and shared by all layers."""

    def __init__(self, pos: torch.Tensor):
        self.pos = pos                                              # (B,T) long
        self.seg_start = pos == 0                                   # (B,T) bool
        self.doc_id = self.seg_start.long().cumsum(1)               # (B,T) row-local segment index
        B, T = pos.shape
        self.doc_start = torch.arange(T, device=pos.device).unsqueeze(0) - pos   # (B,T) first token of the segment
        self._cache: dict = {}

    def causal_mask(self) -> torch.Tensor:
        """(B,T,T) bool: key j visible to query i iff j <= i and same segment."""
        if "causal" not in self._cache:
            T = self.pos.shape[1]
            tri = torch.ones(T, T, dtype=torch.bool, device=self.pos.device).tril()
            self._cache["causal"] = tri.unsqueeze(0) & (self.doc_id.unsqueeze(2) == self.doc_id.unsqueeze(1))
        return self._cache["causal"]

    def cu_seqlens(self) -> tuple[torch.Tensor, int]:
        """Segment boundaries over the flattened (B*T) tokens (flash-attn varlen / fla varlen)."""
        if "cu" not in self._cache:
            flat = self.seg_start.reshape(-1)
            starts = torch.nonzero(flat, as_tuple=False).squeeze(1)
            cu = torch.cat([starts, starts.new_tensor([flat.numel()])]).to(torch.int32)
            self._cache["cu"] = (cu, int((cu[1:] - cu[:-1]).max().item()))
        return self._cache["cu"]


@dataclass
class Ctx:
    """Everything a mixer needs besides its input: runtime flags + document structure (or None)."""
    flags: RunFlags
    doc: DocInfo | None = None
    cache: dict = field(default_factory=dict)

    @property
    def pos(self) -> torch.Tensor | None:
        return None if self.doc is None else self.doc.pos

    def attn_mask(self, T: int, device) -> torch.Tensor:
        """(B,T,T) document-causal mask, or (1,T,T) causal mask without document info."""
        if self.doc is not None:
            return self.doc.causal_mask()
        if "causal" not in self.cache:
            self.cache["causal"] = causal_mask(T, device).unsqueeze(0)
        return self.cache["causal"]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.w_down._is_residual_proj = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(g) * u)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Rotate-half RoPE on the LAST `rope_dim` channels of x (..., T, D); cos/sin: (T, rope_dim/2)."""
    D = x.shape[-1]
    xr = x if rope_dim == D else x[..., D - rope_dim:]
    xr32 = xr.float()
    h = rope_dim // 2
    x1, x2 = xr32[..., :h], xr32[..., h:]
    rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)
    if rope_dim == D:
        return rot
    return torch.cat([x[..., : D - rope_dim], rot], dim=-1)


class Rotary(nn.Module):
    """RoPE cache for `dim` rotated channels. Positions may be arbitrary (e.g. compressed-block ends)."""

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        f = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None, inverse: bool = False) -> torch.Tensor:
        """x: (..., T, D) or (B,H,T,D). pos: (T,) or (B,T) long positions (default 0..T-1; per-document
        positions restart at 0). inverse=True rotates by -pos."""
        if pos is None:
            T = x.shape[-2]
            if T > self.cos.shape[0]:
                raise ValueError(f"sequence length {T} exceeds RoPE cache {self.cos.shape[0]}")
            cos, sin = self.cos[:T], self.sin[:T]
        else:
            cos, sin = self.cos[pos], self.sin[pos]
            if pos.dim() == 2 and x.dim() == 4:           # (B,T,r) -> (B,1,T,r) for (B,H,T,D)
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        if inverse:
            sin = -sin
        return apply_rope(x, cos, sin, self.dim)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def causal_mask(T: int, device) -> torch.Tensor:
    return torch.ones(T, T, dtype=torch.bool, device=device).tril()


# --------------------------------------------------------------------------------------
# Statistics helpers (only used on the probe batch, under no_grad)
# --------------------------------------------------------------------------------------
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def tensor_summary(x: torch.Tensor, max_elems: int = 1 << 20) -> dict:
    """mean/std/min/max and 1/50/99 percentiles of a tensor (percentiles on a fixed subsample)."""
    x = x.detach().float().flatten()
    out = {
        "mean": x.mean().item(),
        "std": x.std().item() if x.numel() > 1 else 0.0,
        "min": x.min().item(),
        "max": x.max().item(),
    }
    if x.numel() > max_elems:
        g = torch.Generator(device="cpu").manual_seed(0)
        idx = torch.randint(0, x.numel(), (max_elems,), generator=g).to(x.device)
        x = x[idx]
    q = torch.quantile(x, torch.tensor([0.01, 0.5, 0.99], device=x.device))
    out["p01"], out["p50"], out["p99"] = (float(v) for v in q.tolist())
    return out


@torch.no_grad()
def head_softmax_stats(q: torch.Tensor, k: torch.Tensor, allowed: torch.Tensor, scale: float,
                       sink: torch.Tensor | None = None) -> dict:
    """Per-head statistics of a softmax attention pattern.

    q: (B,H,T,D); k: (B,H,S,D); allowed: bool, broadcastable to (B,T,S).
    sink: optional (H,) logits of an always-present null key (CSA attention sink).
    Returns per-head lists: max pre-softmax logit, entropy (nats), entropy normalised by
    log(#allowed keys) (1 = uniform, 0 = one-hot) and, with a sink, the sink probability mass.
    """
    B, H, T, _ = q.shape
    S = k.shape[2]
    allowed = allowed.expand(B, T, S)
    n_allowed = allowed.sum(-1).float() + (1.0 if sink is not None else 0.0)
    multi = n_allowed > 1
    log_n = n_allowed.clamp(min=2.0).log()
    res: dict = {"max_logit": [], "entropy": [], "norm_entropy": []}
    if sink is not None:
        res["sink_mass"] = []
        a = torch.cat([allowed, allowed.new_ones(B, T, 1)], dim=-1)
    else:
        a = allowed
    for h in range(H):
        s = torch.matmul(q[:, h].float(), k[:, h].float().transpose(-1, -2)) * scale
        s = s.masked_fill(~allowed, float("-inf"))
        res["max_logit"].append(s.amax().item())
        if sink is not None:
            s = torch.cat([s, s.new_full((B, T, 1), float(sink[h]))], dim=-1)
        logp = torch.log_softmax(s, dim=-1)
        p = logp.exp()
        ent = -(p * logp.masked_fill(~a, 0.0)).sum(-1)
        res["entropy"].append(ent.mean().item())
        res["norm_entropy"].append(masked_mean(ent / log_n, multi).item())
        if sink is not None:
            res["sink_mass"].append(p[..., -1].mean().item())
    return res


def summarize_list(v: list[float]) -> dict:
    t = torch.tensor(v, dtype=torch.float64)
    return {"mean": t.mean().item(), "min": t.min().item(), "max": t.max().item()}
