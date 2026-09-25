"""K: Kimi Delta Attention (Kimi Linear, 2025) - gated delta rule with channel-wise forget gate.

Recurrence per head (state S in R^{dk x dv}):
    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T,     o_t = S_t^T q_t
    alpha_t = exp(g_t),  g_t = -exp(A_log) * softplus(W_up W_down x_t + dt_bias)   (per channel, <= 0)
    beta_t  = sigmoid(W_b x_t)                                                     (per head)
q, k, v: linear -> causal depthwise short conv -> SiLU; q, k L2-normalised.
Output: o_proj( RMSNorm_head(o) * sigmoid(W_g_up W_g_down x) )   (low-rank output gate)

Packed documents: the state is reset to 0 at every document-segment start and the short conv never
reaches into the previous segment, so every document is processed exactly as if it were alone.

Backends: `fla` (flash-linear-attention Triton kernel, varlen via cu_seqlens; used on H100 only when
importable AND it reproduces the reference recurrence in a start-up self-check) or `torch` (chunked
WY-form implementation below; exact, fp32, segment resets inside chunks; works on Windows / RTX 3060).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .common import Ctx, RMSNorm, l2norm, tensor_summary


# --------------------------------------------------------------------------------------
# Reference and chunked kernels
# --------------------------------------------------------------------------------------
def kda_recurrent_ref(q, k, v, g, beta, scale, initial_state=None, seg_start=None):
    """Naive token-by-token recurrence (tests / self-check). q,k,g (B,T,H,K), v (B,T,H,V), beta (B,T,H);
    seg_start (B,T) bool: the state is zeroed before processing a segment's first token."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, g, beta = (t.to(dt) for t in (q, k, v, g, beta))
    S = torch.zeros(B, H, K, V, dtype=dt, device=q.device) if initial_state is None else initial_state.to(dt).clone()
    out = []
    for t in range(T):
        if seg_start is not None:
            S = S * (~seg_start[:, t]).to(dt).view(B, 1, 1, 1)
        S = S * g[:, t].exp().unsqueeze(-1)
        kv_mem = (S * k[:, t].unsqueeze(-1)).sum(-2)                         # S^T k : (B,H,V)
        u = beta[:, t].unsqueeze(-1) * (v[:, t] - kv_mem)
        S = S + k[:, t].unsqueeze(-1) * u.unsqueeze(-2)
        out.append((S * (q[:, t] * scale).unsqueeze(-1)).sum(-2))
    return torch.stack(out, dim=1), S


def _intra_chunk(q, k, gc, beta, v, same=None, keep0=None):
    """Per-chunk quantities (all chunks in parallel). Inputs (..., C, *). Returns Aqk, W, U0.

    Aqk[i,j] = sum_c q_i k_j exp(G_i - G_j)   (j <= i, same segment)
    Akk[i,j] = sum_c k_i k_j exp(G_i - G_j)   (j <  i, same segment)
    (I + diag(beta) Akk) U = diag(beta) (V - (exp(G) * K * keep0) S0)   =>  U = U0 - W S0
    Pairwise exponent differences are taken explicitly (always <= 0): stable for any decay.
    same: (...,C,C) bool, positions in the same document segment; keep0: (...,C) bool, positions
    still connected to the chunk's incoming state (no segment start at or before them in the chunk).
    """
    C = q.shape[-2]
    incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    strict = incl.clone().fill_diagonal_(False)
    if same is not None:
        incl, strict = incl & same, strict & same
    decay = (gc.unsqueeze(-2) - gc.unsqueeze(-3)).masked_fill(~incl.unsqueeze(-1), float("-inf")).exp()
    Aqk = torch.einsum("...ik,...ijk,...jk->...ij", q, decay, k)            # decay: (..., i, j, K)
    Akk = torch.einsum("...ik,...ijk,...jk->...ij", k, decay, k).masked_fill(~strict, 0.0)
    L = torch.eye(C, dtype=q.dtype, device=q.device) + beta.unsqueeze(-1) * Akk
    b = beta.unsqueeze(-1)
    kg = k * gc.exp()
    if keep0 is not None:
        kg = kg * keep0.unsqueeze(-1)
    U0 = torch.linalg.solve_triangular(L, b * v, upper=False, unitriangular=True)
    W = torch.linalg.solve_triangular(L, b * kg, upper=False, unitriangular=True)
    return Aqk, W, U0


def kda_chunk_torch(q, k, v, g, beta, scale, initial_state=None, chunk_size=32, seg_start=None,
                    use_checkpoint=True):
    """Exact chunked KDA in PyTorch. q,k,g: (B,T,H,K) (q,k L2-normalised), v: (B,T,H,V), beta: (B,T,H),
    seg_start: optional (B,T) bool document-segment starts (state reset). Returns o (B,T,H,V) and the
    final state (B,H,K,V), both fp32 (fp64 if the inputs are fp64)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    dt = torch.float64 if q.dtype == torch.float64 else torch.float32
    C = chunk_size
    pad = (C - T % C) % C
    q, k, v, g, beta = (t.to(dt) for t in (q, k, v, g, beta))
    q = q * scale
    if pad:  # zero k/v/beta and zero log-decay on padded positions: they leave the state untouched
        q, k, v, g = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v, g))
        beta = F.pad(beta, (0, 0, 0, pad))
        if seg_start is not None:
            seg_start = F.pad(seg_start, (0, pad), value=False)
    N = (T + pad) // C

    def to_chunks(t):  # (B,Tp,H,D) -> (B,H,N,C,D)
        return t.reshape(B, N, C, H, -1).permute(0, 3, 1, 2, 4)

    q, k, v, g = map(to_chunks, (q, k, v, g))
    beta = beta.reshape(B, N, C, H).permute(0, 3, 1, 2)
    gc = g.cumsum(dim=-2)                                                   # within-chunk cumulative log decay
    same = keep0 = last_same = carry = None
    if seg_start is not None:
        sid = seg_start.reshape(B, 1, N, C).long().cumsum(-1)              # segment index inside each chunk
        same = sid.unsqueeze(-1) == sid.unsqueeze(-2)                       # (B,1,N,C,C)
        keep0 = sid == 0                                                    # (B,1,N,C)
        last_same = sid == sid[..., -1:]                                    # same segment as the chunk's last token
        carry = (sid[..., -1] == 0).to(dt)                                  # (B,1,N) no reset in the chunk
    args = (q, k, gc, beta, v, same, keep0)
    if use_checkpoint and torch.is_grad_enabled():
        Aqk, W, U0 = checkpoint(_intra_chunk, *args, use_reentrant=False)
    else:
        Aqk, W, U0 = _intra_chunk(*args)
    g_last = gc[..., -1, :]                                                  # (B,H,N,K)
    q_dec = q * gc.exp()
    k_dec = k * (g_last.unsqueeze(-2) - gc).exp()
    if seg_start is not None:
        q_dec = q_dec * keep0.unsqueeze(-1)
        k_dec = k_dec * last_same.unsqueeze(-1)
    S = torch.zeros(B, H, K, V, dtype=dt, device=q.device) if initial_state is None else initial_state.to(dt)
    outs = []
    for n in range(N):
        U = U0[:, :, n] - W[:, :, n] @ S
        outs.append(q_dec[:, :, n] @ S + Aqk[:, :, n] @ U)
        decay_S = g_last[:, :, n].exp().unsqueeze(-1)
        if carry is not None:
            decay_S = decay_S * carry[:, :, n].view(B, 1, 1, 1)
        S = S * decay_S + k_dec[:, :, n].transpose(-1, -2) @ U
    o = torch.stack(outs, dim=2).reshape(B, H, N * C, V)[:, :, :T].transpose(1, 2)
    return o, S


# --------------------------------------------------------------------------------------
# fla backend with self-check
# --------------------------------------------------------------------------------------
_FLA_CACHE: dict = {}


def _fla_fn():
    try:
        from fla.ops.kda import chunk_kda  # type: ignore
        return chunk_kda
    except Exception:
        return None


def _fla_call(fn, q, k, v, g, beta, scale, cu_seqlens=None):
    """q,k,g (B,T,H,K) ... With cu_seqlens the batch is flattened to one varlen sequence and the returned
    state holds one entry per segment."""
    if cu_seqlens is None:
        q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
        return fn(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=None, output_final_state=True)
    B, T = q.shape[:2]
    flat = lambda t: t.reshape(1, B * T, *t.shape[2:]).contiguous()
    o, S = fn(q=flat(q), k=flat(k), v=flat(v), g=flat(g), beta=flat(beta), scale=scale, initial_state=None,
              output_final_state=True, cu_seqlens=cu_seqlens.long())
    return o.reshape(B, T, *o.shape[2:]), S


def select_kda_backend(requested: str, device: torch.device, varlen: bool) -> tuple[str, str]:
    """Returns (backend, reason). 'auto' uses fla only if it reproduces the reference recurrence
    (including segment resets through cu_seqlens when `varlen`)."""
    if requested == "torch":
        return "torch", "requested"
    if device.type != "cuda":
        if requested == "fla":
            raise RuntimeError("kda.backend=fla requires CUDA")
        return "torch", "no CUDA"
    key = (requested, str(device), varlen)
    if key in _FLA_CACHE:
        return _FLA_CACHE[key]
    fn = _fla_fn()
    if fn is None:
        res = ("torch", "fla not importable")
    else:
        try:
            g0 = torch.Generator(device="cpu").manual_seed(0)
            B, T, H, K = 2, 150, 2, 64
            mk = lambda *s: torch.randn(*s, generator=g0)
            q = l2norm(mk(B, T, H, K)); k = l2norm(mk(B, T, H, K)); v = mk(B, T, H, K)
            g = -F.softplus(mk(B, T, H, K)) * 0.3
            beta = torch.sigmoid(mk(B, T, H))
            seg = torch.zeros(B, T, dtype=torch.bool)
            seg[:, 0] = True
            if varlen:
                seg[0, 37] = seg[0, 90] = seg[1, 120] = True
            ref, S_ref = kda_recurrent_ref(q, k, v, g, beta, K ** -0.5, seg_start=seg if varlen else None)
            dev = lambda t, d=torch.bfloat16: t.to(device=device, dtype=d)
            cu = None
            if varlen:
                starts = torch.nonzero(seg.reshape(-1)).squeeze(1)
                cu = torch.cat([starts, torch.tensor([B * T])]).to(device)
            o, S = _fla_call(fn, dev(q), dev(k), dev(v), dev(g, torch.float32), dev(beta), K ** -0.5, cu)
            if varlen:   # state of each row's last segment
                S = S[seg.sum(1).cumsum(0).to(S.device) - 1]
            err = ((o.float().cpu() - ref).abs().max() / ref.abs().max()).item()
            err_s = ((S.float().cpu() - S_ref).abs().max() / S_ref.abs().max()).item()
            tag = "varlen " if varlen else ""
            res = ("fla", f"{tag}self-check ok (rel err {err:.2e}/{err_s:.2e})") if max(err, err_s) < 5e-2 else \
                ("torch", f"fla {tag}self-check FAILED (rel err {err:.2e}/{err_s:.2e})")
        except Exception as e:  # signature/version mismatch etc.
            res = ("torch", f"fla self-check raised {type(e).__name__}: {e}")
    if requested == "fla" and res[0] != "fla":
        raise RuntimeError(f"kda.backend=fla unusable: {res[1]}")
    _FLA_CACHE[key] = res
    return res


# --------------------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------------------
class ShortConv(nn.Module):
    """Causal depthwise conv1d + SiLU on (B,T,C). With per-token document positions the window never
    reaches into the previous segment."""

    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, padding=kernel - 1, bias=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        T = x.shape[1]
        if pos is None:
            return F.silu(self.conv(x.transpose(1, 2))[..., :T]).transpose(1, 2)
        w = self.conv.weight[:, 0, :].to(x.dtype)                           # (C, K); tap K-1 = lag 0
        Kc = w.shape[1]
        y = x * w[:, Kc - 1]
        for j in range(1, Kc):
            xs = F.pad(x, (0, 0, j, 0))[:, :T]                               # x_{t-j}
            y = y + xs * (pos >= j).unsqueeze(-1).to(x.dtype) * w[:, Kc - 1 - j]
        return F.silu(y)


class KDAMixer(nn.Module):
    layer_code = "K"

    def __init__(self, cfg: ModelConfig, use_rope: bool = True):  # KDA is position-aware by itself
        super().__init__()
        c = cfg.kda
        self.H, self.K = cfg.n_heads, cfg.head_dim
        self.V = cfg.head_dim
        d, HK, HV = cfg.d_model, self.H * self.K, self.H * self.V
        self.q_proj = nn.Linear(d, HK, bias=False)
        self.k_proj = nn.Linear(d, HK, bias=False)
        self.v_proj = nn.Linear(d, HV, bias=False)
        self.q_conv = ShortConv(HK, c.conv_size)
        self.k_conv = ShortConv(HK, c.conv_size)
        self.v_conv = ShortConv(HV, c.conv_size)
        self.f_down = nn.Linear(d, c.gate_rank, bias=False)       # forget gate (channel-wise, low rank)
        self.f_up = nn.Linear(c.gate_rank, HK, bias=False)
        self.b_proj = nn.Linear(d, self.H, bias=False)              # beta (delta write strength)
        self.g_down = nn.Linear(d, c.gate_rank, bias=False)       # output gate (low rank)
        self.g_up = nn.Linear(c.gate_rank, HV, bias=True)
        self.o_norm = RMSNorm(self.V, cfg.norm_eps)
        self.o_proj = nn.Linear(HV, d, bias=False)
        self.o_proj._is_residual_proj = True
        A = torch.empty(self.H).uniform_(c.A_init_min, c.A_init_max)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(HK) * (math.log(c.dt_max) - math.log(c.dt_min)) + math.log(c.dt_min))
        dt = dt.clamp(min=c.dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        self.chunk_size = c.chunk_size
        self.backend = "torch"
        self.last_diag: dict | None = None

    def set_backend(self, backend: str) -> None:
        self.backend = backend

    def reset_special_parameters(self, cfg: ModelConfig) -> None:
        nn.init.zeros_(self.g_up.bias)

    def forward(self, x: torch.Tensor, ctx: Ctx):
        B, T, _ = x.shape
        H, K, V = self.H, self.K, self.V
        pos = ctx.pos
        seg = ctx.doc.seg_start if ctx.doc is not None else None
        q = self.q_conv(self.q_proj(x), pos).view(B, T, H, K)
        k = self.k_conv(self.k_proj(x), pos).view(B, T, H, K)
        v = self.v_conv(self.v_proj(x), pos).view(B, T, H, V)
        q, k = l2norm(q.float()), l2norm(k.float())
        graw = self.f_up(self.f_down(x)).float() + self.dt_bias.float()
        g = -self.A_log.float().exp().view(1, 1, H, 1) * F.softplus(graw.view(B, T, H, K))   # log alpha <= 0
        beta = torch.sigmoid(self.b_proj(x).float())
        scale = K ** -0.5
        if self.backend == "fla":
            cd = torch.bfloat16 if torch.is_autocast_enabled() else torch.float32
            cu = ctx.doc.cu_seqlens()[0] if seg is not None else None
            o, S = _fla_call(_fla_fn(), q.to(cd), k.to(cd), v.to(cd), g, beta.to(cd), scale, cu)
            if cu is not None:                                  # state at the end of each row's last segment
                S = S[seg.sum(1).cumsum(0) - 1]
            o = o.float()
        else:
            with torch.autocast(device_type=x.device.type, enabled=False):
                o, S = kda_chunk_torch(q, k, v.float(), g, beta, scale, chunk_size=self.chunk_size, seg_start=seg)
        gate = torch.sigmoid(self.g_up(self.g_down(x)).float()).view(B, T, H, V)
        o = self.o_norm(o) * gate
        out = self.o_proj(o.reshape(B, T, H * V))
        if ctx.flags.diag:
            self.last_diag = self._diag(g, beta, S, gate)
        return out, {}

    @torch.no_grad()
    def _diag(self, g, beta, S, gate) -> dict:
        alpha = g.exp()
        # half-life (tokens) of each channel's memory: alpha^n = 0.5
        half_life = (math.log(0.5) / g.clamp(max=-1e-6)).clamp(max=1e6)
        state_norm = S.float().flatten(-2).norm(dim=-1)                   # (B,H) Frobenius norm at row end
        return {
            "forget_gate": tensor_summary(alpha),
            "forget_gate_log": tensor_summary(g),
            "forget_gate_per_head_mean": alpha.mean(dim=(0, 1, 3)).tolist(),
            "forget_half_life_median": float(half_life.float().flatten()[:1 << 22].median()),
            "beta": tensor_summary(beta),
            "beta_per_head_mean": beta.mean(dim=(0, 1)).tolist(),
            "state_norm_mean": state_norm.mean().item(),
            "state_norm_max": state_norm.max().item(),
            "state_norm_per_head": state_norm.mean(0).tolist(),
            "output_gate_mean": gate.mean().item(),
        }
