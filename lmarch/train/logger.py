"""Run logging: append-only JSONL files (source of truth) + optional TensorBoard / W&B mirrors
(W&B: see wandb_sink.py).

<run>/metrics/train_steps.jsonl   every step
<run>/metrics/diagnostics.jsonl   every diag.interval steps (nested per-layer + per-type stats)
<run>/metrics/eval.jsonl          every eval.interval steps
<run>/metrics/events.jsonl        resumes, skips, spikes, rollbacks, alerts, errors, checkpoints
Records are never rewritten. After a resume/rollback, steps are re-logged; readers keep the LAST
record per step (see scripts/analyze_runs.py), while the superseded ones remain for forensics.
Non-finite floats are stored as null (the step record carries explicit NaN/Inf flags).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from ..secrets import redact


def _clean(o):
    if isinstance(o, str):
        return redact(o)                     # API keys never reach a log file
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and callable(o.item):
        try:
            return _clean(o.item())
        except Exception:
            return str(o)
    return o


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps(_clean(rec), separators=(",", ":")) + "\n")
        self.f.flush()

    def sync(self) -> None:
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            out[key] = float(v)
    return out


class RunLogger:
    """Only rank 0 writes metrics; every rank may write events tagged with its rank."""

    def __init__(self, run_dir: Path, rank: int, tensorboard: bool = False, wandb_cfg: dict | None = None):
        """wandb_cfg (rank 0): kwargs for WandbSink (lcfg, run_name, config, tags, group, job_type)."""
        self.rank = rank
        self.run_dir = run_dir
        self.segment = 0
        self.main = rank == 0
        m = run_dir / "metrics"
        self.events = JsonlWriter(m / "events.jsonl") if self.main else None
        self.steps = JsonlWriter(m / "train_steps.jsonl") if self.main else None
        self.diags = JsonlWriter(m / "diagnostics.jsonl") if self.main else None
        self.evals = JsonlWriter(m / "eval.jsonl") if self.main else None
        self.tb = None
        self.wandb = None
        if self.main and tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(run_dir / "tensorboard"))
            except Exception as e:
                print(f"[logger] tensorboard unavailable: {e}")
        if self.main and wandb_cfg:
            from .wandb_sink import WandbSink
            self.wandb = WandbSink(run_dir=run_dir, **wandb_cfg)

    def _mirror(self, prefix: str, rec: dict, step: int) -> None:
        if self.tb is None:
            return
        for k, v in flatten(rec, prefix).items():
            self.tb.add_scalar(k, v, step)

    def step(self, rec: dict) -> None:
        if self.main:
            rec = dict(rec, segment=self.segment)
            self.steps.write(rec)
            self._mirror("train", {k: v for k, v in rec.items() if k not in ("reasons",)}, rec["step"])
            if self.wandb is not None:
                self.wandb.train(rec)

    def diag(self, rec: dict) -> None:
        if self.main:
            rec = dict(rec, segment=self.segment)
            self.diags.write(rec)
            self._mirror("diag", {"by_type": rec.get("by_type", {}), "global": rec.get("global", {})}, rec["step"])
            if self.wandb is not None:
                self.wandb.diag(rec)

    def eval(self, rec: dict) -> None:
        if self.main:
            rec = dict(rec, segment=self.segment)
            self.evals.write(rec)
            self._mirror("eval", rec, rec["step"])
            if self.wandb is not None:
                self.wandb.eval(rec)

    def event(self, kind: str, echo: bool = True, **data) -> None:
        rec = {"time": time.time(), "kind": kind, "rank": self.rank, "segment": self.segment, **data}
        if echo and (self.main or kind in ("error", "oom", "nonfinite_forensics")):
            msg = " ".join(f"{k}={v}" for k, v in data.items() if not isinstance(v, (dict, list)))
            print(redact(f"[event][rank{self.rank}] {kind} {msg}"), flush=True)
        if self.main:
            self.events.write(rec)
            if self.wandb is not None:
                self.wandb.event(kind, _clean(data))
        else:  # non-main ranks: separate file to avoid interleaved writes
            w = JsonlWriter(self.run_dir / "metrics" / f"events_rank{self.rank}.jsonl")
            w.write(rec)
            w.close()

    def sync(self) -> None:
        for w in (self.events, self.steps, self.diags, self.evals):
            if w is not None:
                w.sync()
        if self.tb is not None:
            self.tb.flush()

    def close(self, exit_code: int = 0) -> None:
        for w in (self.events, self.steps, self.diags, self.evals):
            if w is not None:
                w.close()
        if self.tb is not None:
            self.tb.close()
        if self.wandb is not None:
            self.wandb.finish(exit_code)
