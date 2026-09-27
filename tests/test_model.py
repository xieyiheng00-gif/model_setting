"""Model-level tests: parameter budget, causality and document isolation of every mixer, sparse/dense
consistency, indexer gradient isolation, chunked loss, diagnostics plumbing."""
import pytest
import torch
import torch.nn.functional as F

from lmarch.config import ARCH_PRESETS, build_config
from lmarch.model import RunFlags, build_model
from lmarch.model.csa import GatedCompressor

TINY = "configs/test/trunk.yaml"


def tiny(arch, **over):
    cfg = build_config(TINY, arch=arch, overrides=[f"{k}={v}" for k, v in over.items()])
    torch.manual_seed(0)
    return build_model(cfg.model).eval(), cfg


def logits(m, x, sparse=True, pos=None):
    with torch.no_grad():
        return m(x, flags=RunFlags(sparse=sparse), pos=pos, return_logits=True)["logits"]


@pytest.mark.parametrize("arch", sorted(ARCH_PRESETS))
def test_param_budget(arch):
    cfg = build_config("configs/stage/trunk.yaml", arch=arch)
    m = build_model(cfg.model)
    pc = m.param_counts()
    print(arch, f"total {pc['total'] / 1e6:.1f}M non-emb {pc['non_embedding'] / 1e6:.1f}M")
    assert pc["embedding"] == 128256 * 768
    assert 84e6 < pc["non_embedding"] < 92e6, pc["non_embedding"]


@pytest.mark.parametrize("arch", sorted(ARCH_PRESETS))
@pytest.mark.parametrize("sparse", [False, True])
def test_causality(arch, sparse):
    """Changing token p must not change any output at positions < p."""
    m, cfg = tiny(arch)
    T, p = 96, 57
    x = torch.randint(0, cfg.model.vocab_size, (2, T))
    x2 = x.clone()
    x2[:, p] = (x2[:, p] + 1) % cfg.model.vocab_size
    a, b = logits(m, x, sparse), logits(m, x2, sparse)
    assert torch.allclose(a[:, :p], b[:, :p], atol=1e-5), (a[:, :p] - b[:, :p]).abs().max()
    assert not torch.allclose(a[:, p:], b[:, p:])


@pytest.mark.parametrize("arch", sorted(ARCH_PRESETS))
@pytest.mark.parametrize("sparse", [False, True])
def test_document_isolation(arch, sparse):
    """Row = [doc A (40 tokens) | doc B (56 tokens)] with per-document positions: doc B's outputs equal
    doc B processed alone, and nothing in doc A leaks into doc B."""
    m, cfg = tiny(arch)
    T, s = 96, 40                                  # s is a multiple of the CSA block size (4)
    x = torch.randint(0, cfg.model.vocab_size, (2, T))
    pos = torch.cat([torch.arange(s), torch.arange(T - s)]).unsqueeze(0).expand(2, T)
    packed = logits(m, x, sparse, pos)
    alone = logits(m, x[:, s:], sparse, torch.arange(T - s).unsqueeze(0).expand(2, T - s))
    assert torch.allclose(packed[:, s:], alone, atol=1e-5), (packed[:, s:] - alone).abs().max()
    x2 = x.clone()
    x2[:, :s] = torch.randint(0, cfg.model.vocab_size, (2, s))
    assert torch.allclose(logits(m, x2, sparse, pos)[:, s:], packed[:, s:], atol=1e-5)


def test_dsa_masked_path_matches_dense_when_everything_is_selected():
    """With top_k = T-1, every query except the last keeps all its causal keys."""
    T = 64
    m, cfg = tiny("dsa", **{"model.dsa.top_k": T - 1})
    x = torch.randint(0, cfg.model.vocab_size, (2, T))
    assert torch.allclose(logits(m, x, True)[:, :T - 1], logits(m, x, False)[:, :T - 1], atol=1e-5)


def test_csa_masked_path_matches_dense_when_everything_is_selected():
    T = 64
    N = T // 4
    m, cfg = tiny("csa", **{"model.csa.top_k": N - 1})
    x = torch.randint(0, cfg.model.vocab_size, (2, T))
    assert torch.allclose(logits(m, x, True)[:, :4 * N - 1], logits(m, x, False)[:, :4 * N - 1], atol=1e-5)


@pytest.mark.parametrize("arch", ["dsa", "csa", "kda_dsa"])
def test_indexer_trained_only_by_kl(arch):
    m, cfg = tiny(arch)
    m.train()
    x = torch.randint(0, cfg.model.vocab_size, (2, 64))
    y = torch.roll(x, -1, 1)
    out = m(x, y, RunFlags(sparse=True, indexer_loss=True))
    assert "indexer_kl" in out and torch.isfinite(out["aux_loss"])
    out["aux_loss"].backward()
    for n, p in m.named_parameters():
        is_idx = ".indexer." in n or ".idx_" in n
        if is_idx:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        else:
            assert p.grad is None or p.grad.abs().max() == 0, n
    m.zero_grad()
    out = m(x, y, RunFlags(sparse=True, indexer_loss=True))
    out["loss_sum"].backward()
    for n, p in m.named_parameters():
        if ".indexer." in n or ".idx_" in n:
            assert p.grad is None or p.grad.abs().max() == 0, n


def test_flops_per_token_matches_the_formula():
    """MFU numerator: 6 x matmul weights (+ tied LM head) + causal attention of F layers, counted by hand."""
    m, cfg = tiny("dense")
    c = cfg.model
    d, f, L, V, T = c.d_model, c.ffn_dim, c.n_layers, c.vocab_size, cfg.train.seq_len
    per_layer = 4 * d * d + 3 * d * f                   # qkv + o_proj + gate/up + down
    assert m.flops_per_token(T) == 6 * (L * per_layer + V * d) + 6 * T * c.n_heads * c.head_dim * L


def test_chunked_loss_equals_full_cross_entropy():
    m, cfg = tiny("dense", **{"train.loss_chunk_tokens": 37})
    m.loss_chunk_tokens = 37
    m.train()
    x = torch.randint(0, cfg.model.vocab_size, (2, 64))
    y = torch.roll(x, -1, 1)
    y[:, -1] = -100
    y[0, 10] = -100
    out = m(x, y, RunFlags(), return_logits=True)
    ref = F.cross_entropy(out["logits"].reshape(-1, cfg.model.vocab_size), y.reshape(-1), ignore_index=-100,
                          reduction="sum")
    assert torch.allclose(out["loss_sum"], ref, rtol=1e-5)
    assert out["n_valid"].item() == (y != -100).sum().item()
    out["loss_sum"].backward()   # gradients flow through the checkpointed chunks
    assert m.embed.weight.grad is not None and torch.isfinite(m.embed.weight.grad).all()


def test_compressor_is_causal_and_document_aware():
    torch.manual_seed(0)
    comp = GatedCompressor(16, 8, m=4)
    h = torch.randn(1, 32, 16)
    h2 = h.clone()
    h2[:, 13] += 1.0                        # token 13 lives in block 3 (tokens 12..15)
    a, _ = comp(h)
    b, _ = comp(h2)
    assert torch.allclose(a[:, :3], b[:, :3])
    assert not torch.allclose(a[:, 3], b[:, 3]) and not torch.allclose(a[:, 4], b[:, 4])
    assert torch.allclose(a[:, 5:], b[:, 5:])
    doc = torch.cat([torch.zeros(16), torch.ones(16)]).long().unsqueeze(0)   # doc boundary at block 4
    a, _ = comp(h, doc_id=doc)
    alone, _ = comp(h[:, 16:])
    assert torch.allclose(a[:, 4:], alone, atol=1e-6)   # entries of doc 2 == doc 2 compressed alone


@pytest.mark.parametrize("arch", sorted(ARCH_PRESETS))
def test_diag_forward_populates_stats(arch):
    m, cfg = tiny(arch)
    x = torch.randint(0, cfg.model.vocab_size, (2, 64))
    pos = torch.cat([torch.arange(24), torch.arange(40)]).unsqueeze(0).expand(2, 64)
    with torch.no_grad():
        m(x, torch.roll(x, -1, 1), RunFlags(sparse=True, diag=True), pos)
    for b in m.blocks:
        assert b.last_diag and "act_rms" in b.last_diag
        d = b.mixer.last_diag
        assert d
        if b.code in ("F", "S", "C"):
            assert len(d["max_logit"]) == cfg.model.n_heads and len(d["entropy"]) == cfg.model.n_heads
        if b.code in ("S", "C"):
            for k in ("recall_mean", "effective_sparsity", "indexer_kl_dense", "density"):
                assert k in d, k
        if b.code == "K":
            for k in ("forget_gate", "beta", "state_norm_max"):
                assert k in d, k
