"""End-to-end CPU tests on synthetic packed data: exact resume after a crash, NaN rollback, the
stage-1 -> stage-2 branch point (both branches), divergence exit, config errors."""
import json
from pathlib import Path

from lmarch.train.trainer import EXIT_CONFIG, EXIT_ERROR, EXIT_OK, main as train_main


def run(tmp, packed, stage, name, arch="kda_dsa", extra=()):
    args = ["--config", f"configs/test/{stage}.yaml", "--arch", arch, "--run_name", name, "--set",
            f"data.packed_dir={packed.as_posix()}", f"log.out_dir={Path(tmp).as_posix()}", *extra]
    return train_main(args)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def steps(tmp, name):
    last = {}
    for r in jl(Path(tmp) / name / "metrics" / "train_steps.jsonl"):
        last[r["step"]] = r
    return last


def events(tmp, name):
    return jl(Path(tmp) / name / "metrics" / "events.jsonl")


def test_resume_after_crash_is_exact(tmp_path, packed):
    assert run(tmp_path, packed, "trunk", "ref") == EXIT_OK
    # crash at step 12 -> emergency checkpoint -> the restart resumes and reproduces the reference exactly
    assert run(tmp_path, packed, "trunk", "crash", extra=["debug.inject_exception_step=12"]) == EXIT_ERROR
    assert run(tmp_path, packed, "trunk", "crash", extra=["debug.inject_exception_step=12"]) == EXIT_OK
    ref, cr = steps(tmp_path, "ref"), steps(tmp_path, "crash")
    assert sorted(ref) == sorted(cr) == list(range(16))
    for s in range(16):
        assert abs(ref[s]["loss"] - cr[s]["loss"]) < 1e-5, (s, ref[s]["loss"], cr[s]["loss"])
        assert ref[s]["data_index"] == cr[s]["data_index"] == s
    kinds = [e["kind"] for e in events(tmp_path, "crash")]
    assert "error" in kinds and "resume" in kinds
    diag = jl(tmp_path / "crash" / "metrics" / "diagnostics.jsonl")
    assert {"kda", "dsa"} <= set(diag[-1]["by_type"])
    ev = jl(tmp_path / "crash" / "metrics" / "eval.jsonl")
    assert ev[-1]["full"] and len(ev[-1]["val_by_group"]) == 7
    assert "loss_by_source" in ref[3] and ref[3]["valid_tokens"] > 0


def test_nan_batches_trigger_skip_then_rollback_and_recover(tmp_path, packed):
    extra = ["debug.inject_nan_steps=[11,12,13]", "guard.max_consecutive_bad=3", "guard.rollback_skip_extra=2"]
    assert run(tmp_path, packed, "trunk", "nan", extra=extra) == EXIT_OK
    ev = events(tmp_path, "nan")
    rb = [e for e in ev if e["kind"] == "rollback"]
    assert len(rb) == 1 and rb[0]["from_step"] == 13 and rb[0]["to_step"] == 10
    assert rb[0]["next_data_index"] == 16                     # batches 10..15 skipped
    assert any(e["kind"] == "nonfinite_forensics" for e in ev)
    st = steps(tmp_path, "nan")
    assert max(st) == 15 and all(not st[s]["loss_nonfinite"] for s in range(10, 16))
    assert list((tmp_path / "nan" / "crash_reports").glob("*.json"))


def test_branch_point_both_branches(tmp_path, packed):
    """trunk (16 steps) -> 16K branch and 4K branch: same start step, same stage-2 batches, LR decays."""
    for arch in ("dense", "kda_dsa"):
        assert run(tmp_path, packed, "trunk", f"{arch}_trunk_test", arch=arch) == EXIT_OK
        for br in ("s2_16k", "s2_4k"):
            assert run(tmp_path, packed, br, f"{arch}_{br}_test", arch=arch) == EXIT_OK
        s16, s4 = steps(tmp_path, f"{arch}_s2_16k_test"), steps(tmp_path, f"{arch}_s2_4k_test")
        assert min(s16) == min(s4) == 16 and max(s16) == max(s4) == 23
        assert [s16[k]["data_batch"] for k in sorted(s16)] == [s4[k]["data_batch"] for k in sorted(s4)] \
            == list(range(8))
        assert s16[23]["lr_frac"] < 0.2 and s16[16]["lr_frac"] == 1.0
        bi = [e for e in events(tmp_path, f"{arch}_s2_16k_test") if e["kind"] == "branch_init"][0]
        assert bi["step"] == 16 and bi["new_params"] == [] and bi["optimizer_state_copied"] == bi["optimizer_params"]


def test_branch_can_add_an_indexer(tmp_path, packed):
    """dense trunk -> DSA branch (V3.2-style conversion): new indexer params get fresh init + fresh Adam
    state, and the dense warm-up restarts at the branch point."""
    assert run(tmp_path, packed, "trunk", "dense_trunk_test", arch="dense") == EXIT_OK
    assert run(tmp_path, packed, "s2_4k", "dsa_from_dense", arch="dsa",
               extra=["checkpoint.init_from=" + (tmp_path / "dense_trunk_test").as_posix()]) == EXIT_OK
    bi = [e for e in events(tmp_path, "dsa_from_dense") if e["kind"] == "branch_init"][0]
    assert bi["new_params"] and all(".indexer." in n for n in bi["new_params"])
    st = steps(tmp_path, "dsa_from_dense")
    assert st[16]["sparse_phase"] is False and st[20]["sparse_phase"] is True   # warm-up of 4 steps from 16


def test_divergence_exhausts_rollbacks(tmp_path, packed):
    extra = ["debug.inject_nan_steps=[" + ",".join(str(i) for i in range(6, 60)) + "]", "guard.max_rollbacks=2",
             "guard.rollback_skip_extra=0"]
    assert run(tmp_path, packed, "trunk", "div", extra=extra) == 4
    assert any(e["kind"] == "diverged" for e in events(tmp_path, "div"))


def test_bad_config_exit_code(tmp_path, packed):
    assert run(tmp_path, packed, "trunk", "bad", extra=["train.micro_batch_size=7"]) == EXIT_CONFIG
