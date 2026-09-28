"""The model's elementwise work as pure functions that torch.compile can fuse (train.compile_ops).

RMSNorm in fp32, the residual add before a norm, SwiGLU, RoPE, the KDA short conv / gates / q-k L2 norm / gated
output norm, and the softmax of the fused LM-head loss are chains of memory-bound elementwise ops and reductions.
Eager PyTorch runs one kernel per op and reads and writes the whole activation each time. Each function below is
exact eager PyTorch; with compile_ops on, `enable()` swaps in torch.compile'd versions (TorchInductor -> Triton) that
fuse each chain into one or two kernels. Numerics are unchanged apart from summation order.

`self_check(device)` compares compiled and eager outputs and gradients on small inputs at start-up; the caller keeps
eager if anything differs or fails to compile. Shapes may change later (train / eval / probe): the first recompile
turns the changed dimensions dynamic, so there is no recompile per shape after that.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_EAGER: dict = {}
_ACTIVE: dict = {}


def fusable(fn):
    """Register `fn`; calls go through whichever version (eager / compiled) is active."""
    name = fn.__name__
    _EAGER[name] = _ACTIVE[name] = fn

    def dispatch(*args, **kwargs):
        return _ACTIVE[name](*args, **kwargs)

    dispatch.__name__, dispatch.__doc__, dispatch.eager = name, fn.__doc__, fn
    return dispatch


# --------------------------------------------------------------------------------------
# the functions (exact eager PyTorch)
# --------------------------------------------------------------------------------------
def _rms(x, weight, eps: float):
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float()).to(dtype)


@fusable
def rms_norm(x, weight, eps: float):
    """RMSNorm computed in fp32, returned in the input dtype."""
    return _rms(x, weight, eps)


@fusable
def add_rms_norm(x, h, weight, eps: float):
    """(x + h, RMSNorm(x + h)): a block's residual add fused with the norm that reads its result."""
    s = x + h
    return s, _rms(s, weight, eps)


@fusable
def swiglu(gu):
    """SiLU(gate) * up, from the fused gate/up projection output."""
    g, u = gu.chunk(2, dim=-1)
    return F.silu(g) * u


@fusable
def apply_rope(x, cos, sin, rope_dim: int):
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


@fusable
def short_conv_silu(x, w, pos):
    """Causal depthwise conv + SiLU on (B,T,C) with per-document positions: a tap never reaches into the previous
    document. w (C, taps), tap taps-1 = lag 0; pos (B,T) restarts at 0 in every document."""
    T, taps = x.shape[1], w.shape[1]
    y = x * w[:, taps - 1]
    for j in range(1, taps):
        xs = F.pad(x, (0, 0, j, 0))[:, :T]                                  # x_{t-j}
        y = y + xs * (pos >= j).unsqueeze(-1).to(x.dtype) * w[:, taps - 1 - j]
    return F.silu(y)


@fusable
def l2norm_qk(q, k, out_dtype: torch.dtype, eps: float = 1e-6):
    """L2-normalise q and k over the head dim in fp32, return them in out_dtype."""
    qf, kf = q.float(), k.float()
    qn = qf * torch.rsqrt(qf.pow(2).sum(-1, keepdim=True) + eps)
    kn = kf * torch.rsqrt(kf.pow(2).sum(-1, keepdim=True) + eps)
    return qn.to(out_dtype), kn.to(out_dtype)


@fusable
def kda_gates(graw, dt_bias, A_log, braw, H: int, K: int):
    """KDA log forget gate g = -exp(A_log) * softplus(graw + dt_bias) (per channel, <= 0), and beta = sigmoid(braw)."""
    B, T = graw.shape[0], graw.shape[1]
    gr = graw.float() + dt_bias.float()
    g = -A_log.float().exp().view(1, 1, H, 1) * F.softplus(gr.view(B, T, H, K))
    return g, torch.sigmoid(braw.float())


@fusable
def gated_rms_norm(o, weight, eps: float, gate_raw):
    """KDA output: RMSNorm over the head dim (fp32) times sigmoid(gate). o (B,T,H,V); gate_raw (B,T,H*V)."""
    B, T, H, V = o.shape
    return _rms(o.float(), weight, eps) * torch.sigmoid(gate_raw.float()).view(B, T, H, V)


@fusable
def ce_chunk_grad(logits, y):
    """Per-token cross-entropy (fp32) of a (C, V) logits chunk and d(sum of it)/d logits = softmax - onehot, in the
    logits dtype; ignored targets (-100) get 0 loss and 0 gradient."""
    lsm = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    valid = y != -100
    yc = y.clamp(min=0)
    loss = -lsm.gather(1, yc[:, None])[:, 0] * valid
    onehot = torch.arange(lsm.shape[1], device=lsm.device)[None, :] == yc[:, None]
    grad = (lsm.exp() - onehot.to(lsm.dtype)) * valid[:, None].to(lsm.dtype)
    return loss, grad.to(logits.dtype)


# --------------------------------------------------------------------------------------
# switching and the start-up check
# --------------------------------------------------------------------------------------
def compile_available(device: torch.device) -> tuple[bool, str]:
    if device.type != "cuda":
        return False, "not on CUDA"
    if sys.platform == "win32":
        return False, "torch.compile/Triton not used on Windows"
    try:
        import triton  # noqa: F401
    except ImportError:
        return False, "triton not installed"
    return True, "ok"


def enable() -> None:
    import torch._dynamo
    dc = torch._dynamo.config
    for key in ("recompile_limit", "cache_size_limit"):      # renamed across PyTorch versions
        if hasattr(dc, key):
            setattr(dc, key, max(getattr(dc, key), 64))
    for name, fn in _EAGER.items():
        _ACTIVE[name] = torch.compile(fn, fullgraph=True)


def disable() -> None:
    _ACTIVE.update(_EAGER)


def is_enabled() -> bool:
    return any(_ACTIVE[n] is not _EAGER[n] for n in _EAGER)


def _cases(dev):
    """Small inputs for every function: (name, args, indices of args that need gradients)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    r = lambda *s, dt=torch.float32: torch.randn(*s, generator=g).to(dev, dt)
    bf = torch.bfloat16
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8] * 4] * 2, device=dev)
    B, T, H, K = 2, 64, 4, 32
    y = torch.randint(0, 997, (64,), generator=g).to(dev)
    y[::7] = -100
    return [
        ("rms_norm", (r(B, T, 96), r(96), 1e-6), (0, 1)),
        ("add_rms_norm", (r(B, T, 96), r(B, T, 96, dt=bf), r(96), 1e-6), (0, 1, 2)),
        ("swiglu", (r(B, T, 192, dt=bf),), (0,)),
        ("apply_rope", (r(B, H, T, 32, dt=bf), r(T, 8), r(T, 8), 16), (0,)),
        ("short_conv_silu", (r(B, T, 96, dt=bf), r(96, 4, dt=bf), pos), (0, 1)),
        ("l2norm_qk", (r(B, T, H, K, dt=bf), r(B, T, H, K, dt=bf), bf), (0, 1)),
        ("kda_gates", (r(B, T, H * K, dt=bf), r(H * K), r(H), r(B, T, H, dt=bf), H, K), (0, 1, 2, 3)),
        ("gated_rms_norm", (r(B, T, H, K, dt=bf), r(K), 1e-6, r(B, T, H * K, dt=bf)), (0, 1, 3)),
        ("ce_chunk_grad", (r(64, 997, dt=bf), y), ()),
    ]


def _run(fn, args, grad_idx, seed_out: int):
    args = [a.detach().clone().requires_grad_(i in grad_idx) if torch.is_tensor(a) and a.is_floating_point() else a
            for i, a in enumerate(args)]
    out = fn(*args)
    outs = out if isinstance(out, tuple) else (out,)
    grads = []
    if grad_idx:
        g = torch.Generator(device="cpu").manual_seed(seed_out)
        loss = sum((o.float() * torch.randn(o.shape, generator=g).to(o.device)).sum() for o in outs)
        loss.backward()
        grads = [args[i].grad for i in grad_idx]
    return [o.detach().float() for o in outs], [x.float() for x in grads]


def self_check(device: torch.device) -> tuple[bool, str]:
    """Compiled vs eager on small inputs (outputs and input gradients). Call after enable()."""
    worst = 0.0
    for name, args, grad_idx in _cases(device):
        try:
            ref = _run(_EAGER[name], args, grad_idx, 1)
            got = _run(_ACTIVE[name], args, grad_idx, 1)
        except Exception as e:  # noqa: BLE001 - a compile error: keep eager
            return False, f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:200]}"
        for a, b in zip(ref[0] + ref[1], got[0] + got[1]):
            err = ((a - b).abs().max() / a.abs().max().clamp(min=1e-6)).item()
            worst = max(worst, err)
            if not err < 2e-2:
                return False, f"{name}: compiled differs from eager (rel err {err:.2e})"
    return True, f"self-check ok ({len(_EAGER)} functions, worst rel err {worst:.1e})"


def configure(device: torch.device, want: bool) -> str:
    """Turn compiled kernels on (after a self-check) or off; returns a one-line status for the run metadata."""
    disable()
    if not want:
        return "off"
    ok, why = compile_available(device)
    if not ok:
        return f"off ({why})"
    if os.environ.get("LMARCH_NO_COMPILE"):
        return "off (LMARCH_NO_COMPILE set)"
    try:
        enable()
        ok, why = self_check(device)
    except Exception as e:  # noqa: BLE001 - any compile set-up problem: stay eager
        ok, why = False, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    if not ok:
        disable()
        return f"off (self-check failed: {why})"
    return f"on ({why})"
