"""Smoke-test mode: the full two-stage plan on a small slice of the real data, then automatic checks.

  python scripts/make_smoke_data.py                 # once: data/packed_smoke (~40 MB subset of the packed data)
  python scripts/smoke.py                           # all 5 archs: trunk (40 steps) -> 16K branch + 4K branch
  python scripts/smoke.py --archs dense kda_dsa     # a subset
  python scripts/smoke.py --faults                  # + rehearse crash handling on the first arch
  python scripts/smoke.py --config_dir configs/rtx3060 --archs dense   # full-size model memory check

Every run goes through scripts/supervise.py, exactly like a real run. Checks after the runs:
  * every stage finished (exit 0) and wrote per-step, diagnostics and eval records;
  * both branches started from the trunk's final checkpoint at the same step (branch_init event);
  * both branches consumed the same stage-2 batches in the same order;
  * the LR decayed to ~0 at the end of each branch; no NaN/Inf steps (unless --faults);
  * per-type diagnostics exist for every layer type of the architecture;
  * W&B did not fail to start (unless --set log.wandb=false).
Notes (printed, not failures): a CUDA run with document masking that fell back to SDPA with a mask
(flash-attn not installed: slow F layers) or to the PyTorch KDA kernel (fla missing / self-check failed).
With --faults the first arch's trunk also gets poisoned batches (NaN -> skip -> rollback) and one
injected crash (exception -> supervisor restart -> exact resume).
Uses the model/data sizes of <config_dir>/{trunk,s2_16k,s2_4k}.yaml (default configs/smoke).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHS = ["dense", "dsa", "kda_full", "kda_dsa", "csa"]
TYPES = {"dense": {"softmax"}, "dsa": {"dsa"}, "kda_full": {"kda", "softmax"}, "kda_dsa": {"kda", "dsa"},
         "csa": {"csa"}}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def last_per_step(recs):
    d = {}
    for r in recs:
        d[r["step"]] = r
    return [d[k] for k in sorted(d)]


def run_stage(cfg: Path, arch: str, out_dir: Path, hw: str, stage: str, extra: list[str]) -> tuple[int, Path]:
    run_dir = out_dir / f"{arch}_{stage}_{hw}"
    cmd = [sys.executable, str(ROOT / "scripts" / "supervise.py"), "--run_dir", str(run_dir),
           "--backoff_base", "2", "--heartbeat_timeout", "900", "--",
           sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg), "--arch", arch,
           "--set", f"log.out_dir={out_dir.as_posix()}", *extra]
    print(f"\n=== {arch} / {stage} ===\n{' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=ROOT)
    print(f"=== {arch} / {stage}: exit {rc} in {time.time() - t0:.0f}s", flush=True)
    return rc, run_dir


def notes_for(runs: dict) -> list[str]:
    """Kernel fallbacks that make a real run slow (not failures: expected on CPU / Windows / RTX 3060)."""
    out = set()
    for stage, (rc, rd) in runs.items():
        try:
            meta = json.loads((rd / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rt, cuda = meta.get("runtime", {}), meta.get("env", {}).get("device", "cpu") != "cpu"
        if cuda and rt.get("attn_backend") == "sdpa_mask":
            out.add("F/S attention ran on SDPA with a (B,T,T) mask: install flash-attn for real runs (README)")
        if cuda and rt.get("kda_backend") == "torch":
            out.add(f"KDA ran on the PyTorch kernel: {rt.get('kda_backend_reason', '')}")
    return sorted(out)


def check(arch: str, runs: dict, faults: bool) -> list[str]:
    errs = []
    for stage, (rc, rd) in runs.items():
        wb = [e for e in read_jsonl(rd / "metrics" / "events.jsonl") if e.get("kind") == "wandb"]
        if wb and wb[-1].get("mode") == "failed":
            errs.append(f"{stage}: W&B failed to start ({wb[-1].get('error')})")
        if rc != 0:
            errs.append(f"{stage}: exit code {rc}")
            continue
        steps = last_per_step(read_jsonl(rd / "metrics" / "train_steps.jsonl"))
        diag = read_jsonl(rd / "metrics" / "diagnostics.jsonl")
        evals = read_jsonl(rd / "metrics" / "eval.jsonl")
        if not steps or not diag or not evals:
            errs.append(f"{stage}: missing metrics (steps={len(steps)} diag={len(diag)} eval={len(evals)})")
            continue
        bad = [s["step"] for s in steps if s["loss_nonfinite"] or s["grad_nonfinite"]]
        if bad and not (faults and stage == "trunk"):
            errs.append(f"{stage}: non-finite steps {bad}")
        types = set(diag[-1].get("by_type", {}))
        if not TYPES[arch] <= types:
            errs.append(f"{stage}: diagnostics by_type {types} missing {TYPES[arch] - types}")
        if not any(e.get("full") for e in evals):
            errs.append(f"{stage}: no final full holdout eval")
        if stage != "trunk":
            ev = read_jsonl(rd / "metrics" / "events.jsonl")
            bi = [e for e in ev if e["kind"] == "branch_init"]
            if not bi:
                errs.append(f"{stage}: no branch_init event")
            if steps[-1]["lr_frac"] > 0.05:
                errs.append(f"{stage}: LR did not decay (lr_frac {steps[-1]['lr_frac']:.3f} at the last step)")
    b16, b4 = runs.get("s2_16k"), runs.get("s2_4k")
    if b16 and b4 and b16[0] == 0 and b4[0] == 0:
        s16 = last_per_step(read_jsonl(b16[1] / "metrics" / "train_steps.jsonl"))
        s4 = last_per_step(read_jsonl(b4[1] / "metrics" / "train_steps.jsonl"))
        if [s["data_batch"] for s in s16] != [s["data_batch"] for s in s4]:
            errs.append("16K and 4K branches read different stage-2 batches")
        if s16[0]["step"] != s4[0]["step"]:
            errs.append(f"branches start at different steps ({s16[0]['step']} vs {s4[0]['step']})")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    ap.add_argument("--stages", nargs="+", default=["trunk", "s2_16k", "s2_4k"], choices=["trunk", "s2_16k", "s2_4k"])
    ap.add_argument("--config_dir", default="configs/smoke")
    ap.add_argument("--out_dir", default="runs_smoke")
    ap.add_argument("--keep", action="store_true", help="keep previous smoke runs (default: start clean)")
    ap.add_argument("--faults", action="store_true", help="inject NaN batches + a crash into the first trunk run")
    ap.add_argument("--set", nargs="*", default=[], help="extra config overrides for every run")
    a = ap.parse_args()
    cdir, out = ROOT / a.config_dir, ROOT / a.out_dir
    hw = "smoke" if Path(a.config_dir).name == "smoke" else Path(a.config_dir).name
    if not (ROOT / "data" / "packed_smoke" / "trunk_4k").exists():
        print("data/packed_smoke not found: run  python scripts/make_smoke_data.py  first", file=sys.stderr)
        return 2
    if out.exists() and not a.keep:
        shutil.rmtree(out)
    results, failures, notes = {}, {}, {}
    for i, arch in enumerate(a.archs):
        runs = {}
        for stage in a.stages:
            extra = ["--set", *a.set] if a.set else []
            if a.faults and i == 0 and stage == "trunk":
                extra += ["--set", "debug.inject_nan_steps=[12,13,14]", "debug.inject_exception_step=25"]
            runs[stage] = run_stage(cdir / f"{stage}.yaml", arch, out, hw, stage, extra)
            if stage == "trunk" and runs[stage][0] != 0:
                break
        results[arch] = runs
        errs = check(arch, runs, a.faults)
        if a.faults and i == 0 and "trunk" in runs and runs["trunk"][0] == 0:
            ev = read_jsonl(runs["trunk"][1] / "metrics" / "events.jsonl")
            kinds = [e["kind"] for e in ev]
            for k in ("rollback", "nonfinite_forensics", "error", "resume"):
                if k not in kinds:
                    errs.append(f"--faults: expected a '{k}' event in the trunk run")
        failures[arch] = errs
        notes[arch] = notes_for(runs)
    print("\n================ smoke summary ================")
    for arch, runs in results.items():
        status = "OK " if not failures[arch] else "FAIL"
        codes = " ".join(f"{s}={rc}" for s, (rc, _) in runs.items())
        print(f"{status} {arch:9s} {codes}")
        for e in failures[arch]:
            print(f"      - {e}")
        for n in notes[arch]:
            print(f"      note: {n}")
    run_dirs = [str(rd) for runs in results.values() for _, rd in runs.values() if (rd / "metrics").exists()]
    if run_dirs:
        subprocess.call([sys.executable, str(ROOT / "scripts" / "analyze_runs.py"), *run_dirs,
                         "--out", str(ROOT / "reports" / f"smoke_{hw}")], cwd=ROOT)
    return 0 if all(not f for f in failures.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
