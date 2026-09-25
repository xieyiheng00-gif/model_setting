"""Schedule, packed-data reader (vs the TRAINING_DATA.md reference), guard, config."""
import numpy as np
import pytest

from lmarch.config import ConfigError, GuardConfig, ScheduleConfig, build_config
from lmarch.train.guard import TrainingGuard
from lmarch.train.optim import lr_multiplier


def test_wsd_schedule_two_stages():
    trunk = ScheduleConfig(warmup_steps=10, decay_frac=0.0, decay_start=-1, decay_shape="linear")
    assert lr_multiplier(0, 100, trunk) == pytest.approx(0.1)
    assert lr_multiplier(9, 100, trunk) == pytest.approx(1.0)
    assert lr_multiplier(99, 100, trunk) == 1.0                    # stage 1 never decays
    branch = ScheduleConfig(warmup_steps=10, decay_start=100, decay_shape="linear")
    assert lr_multiplier(100, 140, branch) == pytest.approx(1.0)   # branch point
    assert lr_multiplier(120, 140, branch) == pytest.approx(0.5)
    assert lr_multiplier(139, 140, branch) < 0.05
    branch.decay_shape = "1-sqrt"
    assert lr_multiplier(120, 140, branch) == pytest.approx(1 - 0.5 ** 0.5)


def _ref_trunk(L, trunk_dir, R, start):
    """TRAINING_DATA.md reference batcher (verbatim logic)."""
    skip = start * R
    ids, lens, srcs, step = None, [], [], start
    for rows in L.iter_rows(trunk_dir):
        n = len(rows["source"])
        if skip >= n:
            skip -= n
            continue
        new = rows["input_ids"][skip:]
        ids = new if ids is None else np.concatenate([ids, new])
        lens += rows["doc_lens"][skip:]
        srcs += rows["source"][skip:]
        skip = 0
        while len(ids) >= R:
            yield {"input_ids": ids[:R], "doc_lens": lens[:R], "source": srcs[:R], "batch": step}
            ids, lens, srcs, step = ids[R:], lens[R:], srcs[R:], step + 1


@pytest.mark.parametrize("start", [0, 5, 60])
def test_trunk_stream_matches_reference(packed, data_prep, start):
    from lmarch.data.packed import TrunkStream
    ts = TrunkStream(packed / "trunk_4k", 3)
    a_it, b_it = ts.iter_from(start), _ref_trunk(data_prep, packed / "trunk_4k", 3, start)
    for _ in range(3):
        a, b = next(a_it), next(b_it)
        assert a["batch"] == b["batch"] and np.array_equal(a["input_ids"], b["input_ids"])
        assert a["source"] == b["source"]
        assert all(np.array_equal(x, y) for x, y in zip(a["doc_lens"], b["doc_lens"]))


def _pdata(packed, stage, view, seq, rows, micro, world=1, rank=0):
    from lmarch.data.packed import PackedData
    cfg = build_config(f"configs/test/{'trunk' if stage == 'trunk' else 's2_' + view}.yaml",
                       overrides=[f"data.packed_dir={packed.as_posix()}", f"data.max_rows_per_step={rows}"])
    return PackedData(cfg.data, seq, rank, world, micro, 512)


def test_stage2_branches_see_identical_tokens_and_labels(packed):
    from lmarch.data.packed import IGNORE
    p16 = _pdata(packed, "stage2", "16k", 1024, 1, 8)
    p4 = _pdata(packed, "stage2", "4k", 256, 1, 16)
    a = p16.prepare(next(p16.iter_steps(3))[1])
    b = p4.prepare(next(p4.iter_steps(3))[1])
    ta = np.sort(np.concatenate([m["ids"].ravel() for m in a["micro"]]))
    tb = np.sort(np.concatenate([m["ids"].ravel() for m in b["micro"]]))
    assert np.array_equal(ta, tb)
    for st in (a, b):
        for m in st["micro"]:
            ids, y, pos = m["ids"], m["labels"], m["pos"]
            assert ((y[:, :-1] == ids[:, 1:]) | (y[:, :-1] == IGNORE)).all() and (y[:, -1] == IGNORE).all()
            assert ((y[:, :-1] == IGNORE) == (pos[:, 1:] == 0)).all() and (pos[:, 0] == 0).all()


def test_ranks_partition_the_global_batch(packed):
    one = _pdata(packed, "stage2", "4k", 256, 1, 16)
    full = one.prepare(next(one.iter_steps(2))[1])
    rows = np.concatenate([m["ids"] for m in full["micro"]])
    parts = []
    for r in range(4):
        p = _pdata(packed, "stage2", "4k", 256, 1, 4, world=4, rank=r)
        st = p.prepare(next(p.iter_steps(2))[1])
        assert st["global_valid"] == full["global_valid"]        # every rank knows the global count
        parts.append(np.concatenate([m["ids"] for m in st["micro"]]))
    allr = np.concatenate(parts)
    assert sorted(map(bytes, allr)) == sorted(map(bytes, rows))


def test_stream_wraps_after_the_last_step(packed):
    p = _pdata(packed, "trunk", "16k", 256, 0, 8)
    idx, b = next(p.iter_steps(p.n_steps + 1))
    assert idx == p.n_steps + 1 and b["epoch"] == 1 and b["batch"] == 1


def test_guard_skip_then_rollback():
    g = TrainingGuard(GuardConfig(max_consecutive_bad=3, ema_warmup=5))
    for s in range(10):
        assert g.check(s, 3.0 - 0.01 * s, 1.0, True, True).action == "ok"
    assert g.check(10, float("nan"), 1.0, False, True).action == "skip"
    assert g.check(11, float("nan"), 1.0, False, True).action == "skip"
    assert g.check(12, 1.0, float("inf"), True, False).action == "rollback"
    assert not g.is_healthy(15)


def test_guard_spike_detection():
    g = TrainingGuard(GuardConfig(ema_warmup=20, spike_zscore=6, spike_ratio=1.15, spike_rollback_count=2,
                                  spike_window=10))
    rng = np.random.default_rng(0)
    for s in range(50):
        g.check(s, 3.0 + 0.01 * rng.standard_normal(), 1.0, True, True)
    v = g.check(50, 5.0, 1.0, True, True)
    assert v.spike and v.action == "ok"
    assert g.check(51, 5.0, 1.0, True, True).action == "rollback"


def test_configs_and_validation():
    for hw in ("h100x8", "h100x1", "rtx3060", "smoke"):
        for st in ("trunk", "s2_16k", "s2_4k"):
            cfg = build_config(f"configs/{hw}/{st}.yaml", arch="kda_dsa")
            assert cfg.log.run_name == f"kda_dsa_{st}_{hw}"
            if st != "trunk":
                assert cfg.checkpoint.init_from.endswith(f"kda_dsa_trunk_{hw}")
    cfg = build_config("configs/h100x8/s2_16k.yaml", arch="csa")
    assert cfg.train.max_steps == 18770 and cfg.schedule.decay_start == 13351 and cfg.model.max_seq_len == 16384
    with pytest.raises(ConfigError):
        build_config("configs/stage/trunk.yaml", overrides=["model.nonexistent=1"])
    with pytest.raises(ConfigError):
        build_config("configs/stage/trunk.yaml", overrides=["train.seq_len=3000"])   # must divide 4096
