"""Atomic, self-validating checkpoints.

Layout:  <run>/checkpoints/step_0001000/{model.pt, optimizer.pt, optimizer_param_names.json,
                                         trainer_state.json, rng_rank*.pt, COMPLETE}
Write protocol: everything goes to `step_X.tmp/`, COMPLETE is written last, then the directory is
renamed. A checkpoint without COMPLETE (crash mid-save) is ignored and cleaned up. Loading falls back
to older checkpoints if the newest one fails to deserialize.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .dist import DistInfo, barrier

COMPLETE = "COMPLETE"


def _rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def _set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["cuda"])


class CheckpointManager:
    def __init__(self, ckpt_dir: str | Path, d: DistInfo, keep_last: int, milestone_interval: int):
        self.dir = Path(ckpt_dir)
        self.d = d
        self.keep_last, self.milestone = keep_last, milestone_interval
        if d.is_main:
            self.dir.mkdir(parents=True, exist_ok=True)
            for tmp in self.dir.glob("*.tmp"):          # leftovers of an interrupted save
                shutil.rmtree(tmp, ignore_errors=True)
        barrier(d)

    # ---- save ---------------------------------------------------------------------------
    def save(self, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, state: dict,
             tag: str = "") -> Path:
        name = f"step_{step:07d}" + (f"_{tag}" if tag else "")
        final, tmp = self.dir / name, self.dir / (name + ".tmp")
        t0 = time.time()
        if self.d.is_main:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
        barrier(self.d)
        torch.save(_rng_state(), tmp / f"rng_rank{self.d.rank}.pt")
        barrier(self.d)
        if self.d.is_main:
            torch.save(model.state_dict(), tmp / "model.pt")
            torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
            with open(tmp / "optimizer_param_names.json", "w", encoding="utf-8") as f:
                json.dump(getattr(optimizer, "param_names", []), f)
            st = dict(state, step=step, tag=tag, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                      world_size=self.d.world_size)
            with open(tmp / "trainer_state.json", "w", encoding="utf-8") as f:
                json.dump(st, f, indent=2)
            with open(tmp / COMPLETE, "w") as f:
                f.write(str(step))
                f.flush()
                os.fsync(f.fileno())
            if final.exists():
                shutil.rmtree(final)
            os.replace(tmp, final)
            self._retention()
        barrier(self.d)
        self.last_save_seconds = time.time() - t0
        return final

    def _retention(self) -> None:
        ckpts = self.list_valid()
        regular = [(s, p) for s, p in ckpts if not self._is_milestone(s, p)]
        for s, p in regular[self.keep_last:]:
            shutil.rmtree(p, ignore_errors=True)

    def _is_milestone(self, step: int, path: Path) -> bool:
        if path.name.endswith("_final"):
            return True
        return self.milestone > 0 and step % self.milestone == 0 and "_" not in path.name[len("step_"):]

    # ---- discovery / load --------------------------------------------------------------
    def list_valid(self) -> list[tuple[int, Path]]:
        out = []
        for p in self.dir.glob("step_*"):
            if p.is_dir() and not p.name.endswith(".tmp") and (p / COMPLETE).exists():
                try:
                    out.append((int(p.name.split("_")[1]), p))
                except ValueError:
                    continue
        return sorted(out, key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)

    @staticmethod
    def read_state(path: Path) -> dict:
        with open(path / "trainer_state.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def candidates(self, healthy_only: bool = False, max_step: int | None = None) -> list[Path]:
        """Valid checkpoints (newest first), optionally only healthy ones and at/before max_step."""
        out = []
        for step, p in self.list_valid():
            if max_step is not None and step > max_step:
                continue
            try:
                st = self.read_state(p)
            except Exception:
                continue
            if healthy_only and not st.get("healthy", True):
                continue
            out.append(p)
        return out

    def find(self, healthy_only: bool = False, max_step: int | None = None) -> Path | None:
        c = self.candidates(healthy_only, max_step)
        return c[0] if c else None

    def load(self, path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
             restore_rng: bool = True) -> dict:
        sd = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        if optimizer is not None:
            osd = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
            optimizer.load_state_dict(osd)
        rng = path / f"rng_rank{self.d.rank}.pt"
        if restore_rng and rng.exists():  # world size may differ from the saving run: then skip
            _set_rng_state(torch.load(rng, map_location="cpu", weights_only=False))
        return self.read_state(path)

    def load_latest(self, model, optimizer, restore_rng: bool = True) -> tuple[Path, dict] | None:
        """Newest loadable checkpoint; corrupt ones are skipped (and reported by the caller)."""
        self.failed: list[tuple[str, str]] = []
        for _, p in self.list_valid():
            try:
                return p, self.load(p, model, optimizer, restore_rng)
            except Exception as e:  # corrupt file: try the previous checkpoint
                self.failed.append((str(p), f"{type(e).__name__}: {e}"))
        return None

    # ---- branch point -------------------------------------------------------------------
    @staticmethod
    def resolve(path: str | Path) -> Path:
        """A checkpoint dir, or a run dir (-> its newest *_final checkpoint, else its newest checkpoint)."""
        p = Path(path)
        if (p / COMPLETE).exists():
            return p
        ck = p / "checkpoints" if (p / "checkpoints").exists() else p
        valid = sorted([q for q in ck.glob("step_*") if (q / COMPLETE).exists()],
                       key=lambda q: (int(q.name.split("_")[1]), q.name.endswith("_final")), reverse=True)
        finals = [q for q in valid if q.name.endswith("_final")]
        if finals:
            return finals[0]
        if valid:
            return valid[0]
        raise FileNotFoundError(f"no complete checkpoint under {path}")

    def load_branch(self, path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    allow_new: bool) -> tuple[dict, dict]:
        """Start a new stage from another run's checkpoint (the branch point). Weights and Adam moments are
        matched BY NAME: parameters that are new in this model (e.g. a DSA indexer added at the branch)
        keep their fresh init and get fresh optimizer state; parameters that no longer exist are dropped.
        LR/betas/weight decay come from the current config. Returns (trainer_state, info)."""
        sd = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        cur = model.state_dict()
        missing = [k for k in cur if k not in sd]
        unexpected = [k for k in sd if k not in cur]
        mismatch = [k for k in cur if k in sd and sd[k].shape != cur[k].shape]
        if mismatch:
            raise ValueError(f"shape mismatch at the branch point: {mismatch[:5]}")
        if missing and not allow_new:
            raise ValueError(f"parameters missing from the branch checkpoint: {missing[:5]} "
                             "(set checkpoint.init_allow_new_params=true to initialise them fresh)")
        model.load_state_dict({k: v for k, v in sd.items() if k in cur}, strict=False)
        names_file = path / "optimizer_param_names.json"
        old_names = json.loads(names_file.read_text(encoding="utf-8")) if names_file.exists() else []
        osd = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
        by_name = {old_names[int(i)]: st for i, st in osd["state"].items() if int(i) < len(old_names)}
        new_osd = optimizer.state_dict()
        names = getattr(optimizer, "param_names", [])
        new_osd["state"] = {i: by_name[n] for i, n in enumerate(names) if n in by_name}
        optimizer.load_state_dict(new_osd)
        rng = path / f"rng_rank{self.d.rank}.pt"
        if rng.exists():
            _set_rng_state(torch.load(rng, map_location="cpu", weights_only=False))
        info = {"branch_from": str(path), "new_params": missing, "dropped_params": unexpected,
                "optimizer_state_copied": len(new_osd["state"]), "optimizer_params": len(names)}
        return self.read_state(path), info
