"""Monitoring fixes from the 2026-09-26 H100 smoke test: update/weight ratios over weight matrices only,
the ratio-alert grace period, pre-clip per-layer grad norms and spike attribution, W&B start-up failures."""
import sys
import types

import torch

from lmarch.config import DiagConfig, LogConfig, build_config
from lmarch.model import build_model
from lmarch.train.monitor import ParamIndex, RatioAlerts
from lmarch.train.trainer import Trainer


def _model(arch="kda_full"):
    cfg = build_config("configs/test/trunk.yaml", arch=arch)
    torch.manual_seed(0)
    return build_model(cfg.model)


def test_update_ratio_is_over_matrices_and_vectors_are_separate():
    m = _model("kda_full")
    idx = ParamIndex(m)
    snap = idx.snapshot()
    with torch.no_grad():
        for _, p, _, _ in idx.entries:
            p.mul_(1.001 if p.ndim >= 2 else 1.0)        # matrices move by exactly 1e-3, vectors stay put
    st = idx.update_stats(snap)
    kda = [kg for kg in st if kg[1] == "mixer" and idx.layer_types.get(kg[0]) == "kda"]
    assert kda
    for kg in kda:
        s = st[kg]
        assert abs(s["ratio"] - 1e-3) < 1e-6, s               # dt_bias / A_log / conv... do not dilute it
        assert s["vec_ratio"] == 0.0
        all_norm = sum(float(t.double().norm() ** 2) for t, (_, _, k, g) in zip(snap, idx.entries) if (k, g) == kg) ** 0.5
        assert abs(s["w_norm"] - all_norm) / all_norm < 1e-5  # w_norm still covers every parameter
    # a group without matrices (norm gains) falls back to its vectors
    snap = idx.snapshot()
    with torch.no_grad():
        for _, p, _, g in idx.entries:
            if g == "norm":
                p.mul_(1.002)
    st = idx.update_stats(snap)
    norm_groups = [kg for kg in st if kg[1] == "norm"]
    assert norm_groups and all(abs(st[kg]["ratio"] - 2e-3) < 1e-6 for kg in norm_groups)


def test_ratio_alerts_wait_for_alert_start_step():
    a = RatioAlerts(DiagConfig(ratio_persist=2, alert_start_step=300))
    high = {("embed", "embed"): {"ratio": 2e-2}}
    assert a.update(high, 1.0, 100) == []
    assert a.update(high, 1.0, 200) == []
    assert a.update(high, 1.0, 300) == []                   # first counted check
    out = a.update(high, 1.0, 400)
    assert len(out) == 1 and out[0]["state"] == "high" and out[0]["layer"] == "embed"


def test_layer_grad_norms_undo_clipping():
    m = _model("dense")
    idx = ParamIndex(m)
    torch.manual_seed(1)
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    pre = idx.grad_norms()
    gnorm = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
    assert gnorm > 1.0
    ns = types.SimpleNamespace(pindex=idx)
    rec = Trainer._layer_grad_norms(ns, gnorm, 1.0)
    for kg, v in pre.items():
        assert abs(rec[kg] - v) / v < 1e-4, (kg, rec[kg], v)
    assert Trainer._layer_grad_norms(ns, 0.5, 1.0) == idx.grad_norms()   # nothing clipped: unchanged


def test_grad_culprits_name_the_largest_layer():
    ns = types.SimpleNamespace(pindex=types.SimpleNamespace(layer_types={"L00": "kda", "L01": "softmax"}))
    gs = {("L00", "mixer"): 3.0, ("L00", "ffn"): 4.0, ("L01", "mixer"): 96.0, ("embed", "embed"): 1.0}
    d = Trainer._grad_culprits(ns, gs)
    assert d["grad_culprit"] == "L01.mixer(softmax) 96"
    assert list(d["top_grad_norms"])[0] == "L01.mixer(softmax)"
    assert d["grad_norm_by_type"] == {"embed": 1.0, "kda/ffn": 4.0, "kda/mixer": 3.0, "softmax/mixer": 96.0}
    assert Trainer._grad_culprits(ns, {}) == {}


def test_wandb_init_failure_is_recorded(tmp_path, monkeypatch):
    fake = types.ModuleType("wandb")                    # not a package: the runid import falls back to util
    fake.util = types.SimpleNamespace(generate_id=lambda: "abc123")
    fake.Settings = lambda **kw: kw

    def boom(**kw):
        raise RuntimeError("API changed")

    fake.init = boom
    monkeypatch.setitem(sys.modules, "wandb", fake)
    from lmarch.train.wandb_sink import WandbSink
    s = WandbSink(LogConfig(wandb=True, wandb_mode="offline"), tmp_path, "run", {}, [], "g", "trunk")
    assert s.run is None and s.status == "failed" and "API changed" in s.error
