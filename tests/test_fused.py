"""Speed-ups that must not change the maths: the fused LM-head loss (values and gradients equal to chunked_ce),
the fusable elementwise functions (equal to independent reference formulas), compiled vs eager on CUDA, and
activation_checkpointing=auto."""
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from lmarch.config import activation_checkpointing, build_config
from lmarch.model import RunFlags, build_model
from lmarch.model import fused
from lmarch.model.model import chunked_ce, linear_ce

TINY = "configs/test/trunk.yaml"


def _rand(*s, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*s, generator=g)


def test_linear_ce_equals_chunked_ce():
    B, T, d, V = 2, 150, 32, 211
    h0, w0 = _rand(B, T, d, seed=1), 0.1 * _rand(V, d, seed=2)
    y = torch.randint(0, V, (B, T), generator=torch.Generator().manual_seed(3))
    y[:, ::7] = -100

    def reference(h, w):
        tok = chunked_ce(h, w, y, 64)[0]
        return tok.sum(), tok

    out = []
    for fn in (reference, lambda h, w: linear_ce(h, w, y, 64)):
        h, w = h0.clone().requires_grad_(), w0.clone().requires_grad_()
        loss, tok = fn(h, w)
        (3.0 * loss).backward()                          # the trainer scales the sum: backward must follow
        out.append((loss.detach(), tok.detach(), h.grad, w.grad))
    for a, b in zip(*out):
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)


def test_linear_ce_rows_without_targets():
    h, w = _rand(1, 40, 16).requires_grad_(), (0.1 * _rand(50, 16, seed=1)).requires_grad_()
    y = torch.full((1, 40), -100)
    y[0, :10] = 3
    loss, tok = linear_ce(h, w, y, 16)
    loss.backward()
    assert torch.all(tok[0, 10:] == 0) and torch.all(h.grad[0, 10:] == 0)


def test_fusable_functions_match_reference_formulas():
    x, wt = _rand(2, 9, 24), _rand(24, seed=1)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * wt
    torch.testing.assert_close(fused.rms_norm(x, wt, 1e-6), ref)
    hh = _rand(2, 9, 24, seed=2)
    s, n = fused.add_rms_norm(x, hh, wt, 1e-6)
    torch.testing.assert_close(s, x + hh)
    torch.testing.assert_close(n, fused.rms_norm(x + hh, wt, 1e-6))
    gu = _rand(2, 9, 48, seed=3)
    torch.testing.assert_close(fused.swiglu(gu), F.silu(gu[..., :24]) * gu[..., 24:])
    # short conv with document masking, against an explicit per-position sum
    B, T, C, taps = 2, 12, 5, 4
    xc, wc = _rand(B, T, C, seed=4), _rand(C, taps, seed=5)
    pos = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3]] * B)
    exp = torch.zeros(B, T, C)
    for b in range(B):
        for t in range(T):
            for j in range(taps):
                if pos[b, t] >= j:
                    exp[b, t] += xc[b, t - j] * wc[:, taps - 1 - j]
    torch.testing.assert_close(fused.short_conv_silu(xc, wc, pos), F.silu(exp), rtol=1e-5, atol=1e-6)
    q, k = _rand(2, 7, 3, 8, seed=6), _rand(2, 7, 3, 8, seed=7)
    qn, kn = fused.l2norm_qk(q, k, torch.float32)
    torch.testing.assert_close(qn, q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt())
    torch.testing.assert_close(kn, k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt())
    graw, dtb, alog, braw = _rand(2, 7, 24, seed=8), _rand(24, seed=9), _rand(3, seed=10), _rand(2, 7, 3, seed=11)
    g, beta = fused.kda_gates(graw, dtb, alog, braw, 3, 8)
    torch.testing.assert_close(g, -alog.exp().view(1, 1, 3, 1) * F.softplus((graw + dtb).view(2, 7, 3, 8)))
    torch.testing.assert_close(beta, torch.sigmoid(braw))
    o, ow, gr = _rand(2, 7, 3, 8, seed=12), _rand(8, seed=13), _rand(2, 7, 24, seed=14)
    torch.testing.assert_close(fused.gated_rms_norm(o, ow, 1e-6, gr),
                               fused.rms_norm(o, ow, 1e-6) * torch.sigmoid(gr).view(2, 7, 3, 8))
    logits, y = _rand(10, 30, seed=15).requires_grad_(), torch.tensor([1, 2, -100, 4, 5, 6, 7, -100, 9, 0])
    loss, grad = fused.ce_chunk_grad(logits.detach(), y)
    ref = F.cross_entropy(logits, y, ignore_index=-100, reduction="none")
    ref.sum().backward()
    torch.testing.assert_close(loss, ref.detach())
    torch.testing.assert_close(grad, logits.grad)


@pytest.mark.parametrize("arch", ["dense", "kda_full"])
def test_model_loss_and_grads_equal_with_fused_ce(arch):
    cfg = build_config(TINY, arch=arch)
    torch.manual_seed(0)
    x = torch.randint(0, cfg.model.vocab_size, (2, 64))
    y = torch.randint(0, cfg.model.vocab_size, (2, 64))
    y[:, -1] = -100
    pos = torch.cat([torch.arange(40), torch.arange(24)]).repeat(2, 1)
    res = []
    for use in (False, True):
        torch.manual_seed(0)
        m = build_model(cfg.model).train()
        m.configure_runtime(torch.device("cpu"), True, False, fused_ce=use)
        assert m.fused_ce is use
        out = m(x, y, RunFlags(sparse=True), pos)
        out["loss_sum"].backward()
        res.append((out["loss_sum"].detach(), out["row_loss"], {n: p.grad.clone() for n, p in m.named_parameters()
                                                                  if p.grad is not None}))
    torch.testing.assert_close(res[1][0], res[0][0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(res[1][1], res[0][1], rtol=1e-6, atol=1e-5)
    assert res[0][2].keys() == res[1][2].keys()
    for n in res[0][2]:
        torch.testing.assert_close(res[1][2][n], res[0][2][n], rtol=1e-4, atol=1e-6, msg=n)


def test_runtime_switches_report_their_state():
    m = build_model(build_config(TINY, arch="kda_full").model)
    info = m.configure_runtime(torch.device("cpu"), True, False, fused_ce=True, compile_ops=True)
    assert info["fused_ce"].startswith("on (self-check ok") and m.fused_ce
    assert info["compile_ops"] == "off (not on CUDA)" and not fused.is_enabled()


@pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("triton") is None,
                    reason="needs CUDA + Triton")
def test_compiled_ops_match_eager_on_cuda():
    status = fused.configure(torch.device("cuda"), True)
    try:
        assert status.startswith("on (self-check ok"), status
    finally:
        fused.disable()


def test_activation_checkpointing_auto():
    for arch, want in (("dense", False), ("kda_full", False), ("dsa", True), ("kda_dsa", True), ("csa", True)):
        assert activation_checkpointing(build_config("configs/h100x8/s2_16k.yaml", arch=arch)) is want
        assert activation_checkpointing(build_config("configs/h100x8/trunk.yaml", arch=arch)) is False
    assert activation_checkpointing(build_config("configs/rtx3060/s2_16k.yaml", arch="kda_full")) is True
