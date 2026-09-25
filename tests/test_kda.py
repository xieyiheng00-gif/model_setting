"""Chunked KDA kernel == naive recurrence (outputs, final state, gradients), with and without
document-segment resets."""
import pytest
import torch
import torch.nn.functional as F

from lmarch.model.common import l2norm
from lmarch.model.kda import ShortConv, kda_chunk_torch, kda_recurrent_ref


def _inputs(B=2, T=45, H=3, K=8, V=6, strong_decay=False, seed=0):
    g0 = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g0, dtype=torch.float64)
    q, k = l2norm(r(B, T, H, K)), l2norm(r(B, T, H, K))
    v = r(B, T, H, V)
    g = -F.softplus(r(B, T, H, K)) * (8.0 if strong_decay else 0.5)
    beta = torch.sigmoid(r(B, T, H))
    return q, k, v, g, beta


def _segs(B, T, starts):
    s = torch.zeros(B, T, dtype=torch.bool)
    s[:, 0] = True
    for b, t in starts:
        s[b, t] = True
    return s


@pytest.mark.parametrize("T,chunk", [(45, 16), (64, 16), (7, 32), (100, 32)])
@pytest.mark.parametrize("strong_decay", [False, True])
def test_forward_matches_recurrence(T, chunk, strong_decay):
    q, k, v, g, beta = _inputs(T=T, strong_decay=strong_decay)
    ref, S_ref = kda_recurrent_ref(q, k, v, g, beta, 0.3)
    out, S = kda_chunk_torch(q, k, v, g, beta, 0.3, chunk_size=chunk)
    assert torch.allclose(out, ref, atol=1e-9, rtol=1e-7)
    assert torch.allclose(S, S_ref, atol=1e-9, rtol=1e-7)


def test_initial_state_and_split_equivalence():
    q, k, v, g, beta = _inputs(T=60)
    full, S_full = kda_chunk_torch(q, k, v, g, beta, 0.3, chunk_size=16)
    a, S_a = kda_chunk_torch(q[:, :25], k[:, :25], v[:, :25], g[:, :25], beta[:, :25], 0.3, chunk_size=16)
    b, S_b = kda_chunk_torch(q[:, 25:], k[:, 25:], v[:, 25:], g[:, 25:], beta[:, 25:], 0.3,
                             initial_state=S_a, chunk_size=16)
    assert torch.allclose(torch.cat([a, b], 1), full, atol=1e-9)
    assert torch.allclose(S_b, S_full, atol=1e-9)


@pytest.mark.parametrize("strong_decay", [False, True])
def test_segment_resets_match_reference(strong_decay):
    T = 70
    q, k, v, g, beta = _inputs(T=T, strong_decay=strong_decay)
    seg = _segs(2, T, [(0, 17), (0, 32), (0, 33), (0, 50), (1, 31), (1, 64)])   # incl. chunk-aligned starts
    ref, S_ref = kda_recurrent_ref(q, k, v, g, beta, 0.3, seg_start=seg)
    out, S = kda_chunk_torch(q, k, v, g, beta, 0.3, chunk_size=16, seg_start=seg)
    assert torch.allclose(out, ref, atol=1e-9, rtol=1e-7)
    assert torch.allclose(S, S_ref, atol=1e-9, rtol=1e-7)
    # a document inside the packed row == the same document processed alone
    alone, _ = kda_chunk_torch(q[:1, 17:32], k[:1, 17:32], v[:1, 17:32], g[:1, 17:32], beta[:1, 17:32], 0.3,
                               chunk_size=16)
    assert torch.allclose(out[:1, 17:32], alone, atol=1e-9)


def test_gradients_match_recurrence_with_resets():
    T = 37
    q, k, v, g, beta = _inputs(T=T)
    seg = _segs(2, T, [(0, 9), (1, 20)])
    ins = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    ref, S = kda_recurrent_ref(*ins, 0.3, seg_start=seg)
    (ref.pow(2).sum() + S.sum()).backward()
    g_ref = [t.grad.clone() for t in ins]
    ins2 = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    out, S2 = kda_chunk_torch(*ins2, 0.3, chunk_size=16, seg_start=seg)
    (out.pow(2).sum() + S2.sum()).backward()
    for a, b in zip(g_ref, [t.grad for t in ins2]):
        assert torch.allclose(a, b, atol=1e-8, rtol=1e-6)


def test_fp32_path_is_close():
    q, k, v, g, beta = _inputs(T=128, strong_decay=True)
    ref, _ = kda_recurrent_ref(q, k, v, g, beta, 0.3)
    out, _ = kda_chunk_torch(*(t.float() for t in (q, k, v, g, beta)), 0.3, chunk_size=32)
    assert (out.double() - ref).abs().max() < 1e-4


def test_short_conv_document_masking():
    torch.manual_seed(0)
    conv = ShortConv(6, 4).double()
    x = torch.randn(1, 20, 6, dtype=torch.float64)
    pos_plain = torch.arange(20).unsqueeze(0)
    assert torch.allclose(conv(x, pos_plain), conv(x), atol=1e-12)        # masked path == conv1d path
    pos = torch.cat([torch.arange(8), torch.arange(12)]).unsqueeze(0)      # second document starts at 8
    y = conv(x, pos)
    assert torch.allclose(y[:, 8:], conv(x[:, 8:]), atol=1e-12)            # == the document alone
