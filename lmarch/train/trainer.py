"""Training loop with per-step metrics, periodic diagnostics, holdout evaluation, checkpointing,
branch-point initialisation (stage 1 -> stage 2) and crash recovery.

Data: the packed Llama-3 dataset (lmarch/data/packed.py). Step s reads data index
    i = s - stage_start_step + data_skip
of the configured stage (trunk: 128 x 4K rows; stage 2: 32 x 16K rows or the same rows re-cut to
128 x 4K). The loss is averaged over all non-ignored labels of the GLOBAL batch.

Exit codes (read by scripts/supervise.py; also written to <run>/exit_status_rank*.json):
  0 finished | 1 unexpected error (restart) | 2 config/data error (do not restart) | 3 CUDA OOM
  (restart with a smaller micro-batch) | 4 diverged beyond max_rollbacks (needs a human) |
  5 preempted/SIGTERM, checkpoint saved (restart) | 6 user stop (Ctrl+C, do not restart)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import random
import signal
import socket
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .. import __version__
from ..config import ARCH_PRESETS, LAYER_TYPE_NAMES, Config, ConfigError, build_config, save_config, stage_tag
from ..data.packed import N_GROUPS, Holdout, PackedData, Prefetcher, group_name
from ..model import RunFlags, build_model
from ..secrets import load_secrets, redact
from .checkpoint import CheckpointManager
from .dist import DistInfo, all_reduce_, barrier, cleanup, setup_distributed
from .guard import TrainingGuard
from .logger import RunLogger
from .monitor import GROUPS, ParamIndex, RatioAlerts, collect_probe, group_by_type, health
from .optim import build_optimizer, lr_multiplier

EXIT_OK, EXIT_ERROR, EXIT_CONFIG, EXIT_OOM, EXIT_DIVERGED, EXIT_PREEMPTED, EXIT_USER_STOP = 0, 1, 2, 3, 4, 5, 6


class DivergenceError(RuntimeError):
    pass


class StopRequested(Exception):
    def __init__(self, code: int):
        super().__init__(code)
        self.code = code


@dataclass
class TrainerState:
    step: int = 0                  # completed optimizer steps (= index of the next step), absolute
    stage_start_step: int = 0      # absolute step whose data index is 0 in this stage
    data_skip: int = 0             # data batches skipped by rollbacks
    tokens_seen: int = 0
    lr_scale: float = 1.0          # persistent LR multiplier (reduced by rollbacks if configured)
    n_rollbacks: int = 0
    n_skipped: int = 0
    segment: int = 0               # increments on every resume / rollback (tags log records)
    best_val: float | None = None
    indexer_start_step: int = 0    # step at which the DSA/CSA indexers were created (dense warm-up origin)


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _to_dev(t, dev):
    return None if t is None else t.to(dev, non_blocking=True)


class Trainer:
    def __init__(self, cfg: Config, d: DistInfo):
        self.cfg, self.d, self.dev = cfg, d, d.device
        self.run_dir = Path(cfg.log.out_dir) / cfg.log.run_name
        if d.is_main:
            (self.run_dir / "crash_reports").mkdir(parents=True, exist_ok=True)
        barrier(d)
        (self.run_dir / f"exit_status_rank{d.rank}.json").unlink(missing_ok=True)
        self._seed(cfg.train.seed)
        if self.dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = cfg.train.tf32
            torch.backends.cudnn.allow_tf32 = cfg.train.tf32
        wandb_cfg = None
        if cfg.log.wandb and d.is_main:
            tags = [cfg.model.arch, stage_tag(cfg), cfg.hardware or "run", cfg.model.pattern()]
            tags += [str(t) for t in cfg.log.wandb_tags]
            wandb_cfg = {"lcfg": cfg.log, "run_name": cfg.log.run_name, "config": cfg.to_dict(), "tags": tags,
                         "group": cfg.log.wandb_group, "job_type": stage_tag(cfg)}
        self.log = RunLogger(self.run_dir, d.rank, cfg.log.tensorboard, wandb_cfg)
        if self.log.wandb is not None:
            self.log.event("wandb", mode=self.log.wandb.status, url=getattr(self.log.wandb.run, "url", None),
                           group=cfg.log.wandb_group)

        # ---- data -----------------------------------------------------------------------
        T = cfg.train.seq_len
        self.data = PackedData(cfg.data, T, d.rank, d.world_size, cfg.train.micro_batch_size, cfg.model.vocab_size)
        self.holdout = Holdout(Path(cfg.data.packed_dir) / "holdout", self.data.L)
        self.accum = self.data.accum
        self.rows_per_step = self.data.rows_per_step
        self.tokens_per_step = self.data.tokens_per_step
        self.eval_micro = cfg.eval.micro_batch_size or cfg.train.micro_batch_size

        # ---- model / optimizer ------------------------------------------------------------
        self._seed(cfg.train.seed)
        self.raw = build_model(cfg.model).to(self.dev)
        self.raw.grad_checkpointing = cfg.train.activation_checkpointing
        self.raw.z_loss_coef = cfg.train.z_loss_coef
        self.raw.loss_chunk_tokens = cfg.train.loss_chunk_tokens
        self.runtime_info = self.raw.configure_runtime(self.dev, cfg.data.doc_mask, cfg.train.dtype == "bf16")
        model = self.raw
        if cfg.train.compile:
            model = torch.compile(model)
        if d.is_dist:
            model = DDP(model, device_ids=[d.local_rank] if self.dev.type == "cuda" else None,
                        broadcast_buffers=False)
        self.model = model
        self.opt = build_optimizer(self.raw, cfg.optim, cfg.model.indexer.lr_mult, self.dev)
        self.pindex = ParamIndex(self.raw)
        self.alerts = RatioAlerts(cfg.diag)
        self.guard = TrainingGuard(cfg.guard)
        self.state = TrainerState()
        self.has_indexer = bool(self.raw.indexer_layers)
        self.indexer_types = [LAYER_TYPE_NAMES[self.raw.pattern[i]] for i in self.raw.indexer_layers]
        self.ckpt = CheckpointManager(self.run_dir / "checkpoints", d, cfg.checkpoint.keep_last,
                                      cfg.checkpoint.milestone_interval)
        self.prefetch: Prefetcher | None = None
        self.stop_code = 0
        self.forensics_done = 0
        self.rollback_counts: Counter = Counter()
        self.last_ckpt_step = -1
        self.last_heartbeat = 0.0
        self._resume_or_branch()
        if self.state.step >= cfg.train.max_steps and self.last_ckpt_step != self.state.step:
            raise ConfigError(f"start step {self.state.step} >= train.max_steps {cfg.train.max_steps}")
        if d.is_main:
            self._write_meta()
        self.probe = self.holdout.probe(cfg.diag.probe_batch_size, T, cfg.data.doc_mask) if d.is_main else None
        self._install_signals()

    # ==================================================================================
    # setup helpers
    # ==================================================================================
    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _autocast(self):
        return torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16,
                              enabled=self.cfg.train.dtype == "bf16")

    def _install_signals(self) -> None:
        def handler(signum, frame):
            if signum == signal.SIGINT and self.stop_code == EXIT_USER_STOP:
                raise KeyboardInterrupt  # second Ctrl+C: stop immediately
            self.stop_code = EXIT_USER_STOP if signum == signal.SIGINT else EXIT_PREEMPTED
            print(f"[rank{self.d.rank}] signal {signum}: checkpoint + stop after the current step", flush=True)

        for name in ("SIGINT", "SIGTERM", "SIGUSR1", "SIGBREAK"):   # SIGBREAK: Windows supervisor stop
            s = getattr(signal, name, None)
            if s is not None:
                try:
                    signal.signal(s, handler)
                except (ValueError, OSError):
                    pass

    def data_index(self, step: int) -> int:
        return step - self.state.stage_start_step + self.state.data_skip

    def _state_payload(self, healthy: bool) -> dict:
        return {"trainer": asdict(self.state), "guard": self.guard.state_dict(),
                "alerts": self.alerts.state_dict(), "healthy": healthy, "config": self.cfg.to_dict(),
                "data_index_next": self.data_index(self.state.step), "stage": stage_tag(self.cfg)}

    def _apply_loaded_state(self, st: dict) -> None:
        names = {f.name for f in fields(TrainerState)}
        self.state = TrainerState(**{k: v for k, v in st.get("trainer", {}).items() if k in names})
        self.guard.load_state_dict(st.get("guard", {}))
        self.alerts.load_state_dict(st.get("alerts", {}))

    def _resume_or_branch(self) -> None:
        """1) the run's own checkpoints (crash recovery), 2) checkpoint.init_from (branch point of a new
        stage), 3) from scratch."""
        cfg = self.cfg
        mode = cfg.checkpoint.resume
        if mode not in ("auto", "none"):
            path = Path(mode)
            st = self.ckpt.load(path, self.raw, self.opt)
            return self._after_resume(path, st)
        if mode == "auto":
            res = self.ckpt.load_latest(self.raw, self.opt)
            for p, err in getattr(self.ckpt, "failed", []):
                self.log.event("checkpoint_corrupt", path=p, error=err)
            if res is not None:
                return self._after_resume(*res)
        elif self.ckpt.list_valid():
            self.log.event("warning", message="checkpoints exist but checkpoint.resume=none: starting over; "
                                              "older checkpoints may be overwritten")
        if cfg.checkpoint.init_from:
            src = CheckpointManager.resolve(cfg.checkpoint.init_from)
            st, info = self.ckpt.load_branch(src, self.raw, self.opt, cfg.checkpoint.init_allow_new_params)
            step = int(st.get("step", st.get("trainer", {}).get("step", 0)))
            ds = cfg.schedule.decay_start
            if ds >= 0 and step != ds:
                raise ConfigError(f"branch checkpoint {src} is at step {step}, but this stage decays from step {ds}: "
                                  "is the trunk run finished? (use its *_final checkpoint or fix schedule.decay_start)")
            prev = st.get("trainer", {})
            self.state = TrainerState(step=step, tokens_seen=int(prev.get("tokens_seen", 0)),
                                      indexer_start_step=int(prev.get("indexer_start_step", 0)))
            if any(".indexer." in n or ".idx_" in n for n in info["new_params"]):
                self.state.indexer_start_step = step       # new indexers: dense warm-up starts at the branch
            self.state.stage_start_step = cfg.data.stage_start_step if cfg.data.stage_start_step >= 0 else step
            self.log.event("branch_init", step=step, stage=stage_tag(cfg), **info,
                           stage_start_step=self.state.stage_start_step,
                           branch_stage=st.get("stage"), branch_config_arch=st.get("config", {}).get("model", {}).get("arch"))
            return
        self.state.stage_start_step = max(cfg.data.stage_start_step, 0)
        self.log.event("start", fresh=True, stage=stage_tag(cfg))

    def _after_resume(self, path, st) -> None:
        self._apply_loaded_state(st)
        self.state.segment += 1
        self.log.segment = self.state.segment
        self.last_ckpt_step = self.state.step
        self.log.event("resume", path=str(path), step=self.state.step, data_index=self.data_index(self.state.step),
                       n_rollbacks=self.state.n_rollbacks, lr_scale=self.state.lr_scale)
        diff = _dict_diff(st.get("config", {}), self.cfg.to_dict())
        if diff:
            self.log.event("config_changed_on_resume", diff=diff)

    def _write_meta(self) -> None:
        cfg = self.cfg
        steps_left = cfg.train.max_steps - self.state.stage_start_step
        meta = {
            "lmarch_version": __version__, "arch": cfg.model.arch, "pattern": self.raw.pattern,
            "stage": stage_tag(cfg), "layer_types": [LAYER_TYPE_NAMES[c] for c in self.raw.pattern],
            "params": self.raw.param_counts(), "runtime": self.runtime_info,
            "world_size": self.d.world_size, "micro_batch_size": cfg.train.micro_batch_size,
            "grad_accum": self.accum, "rows_per_step": self.rows_per_step, "seq_len": cfg.train.seq_len,
            "tokens_per_step": self.tokens_per_step,
            "stage_start_step": self.state.stage_start_step, "max_steps": cfg.train.max_steps,
            "stage_tokens_planned": steps_left * self.tokens_per_step,
            "data": {"packed_dir": cfg.data.packed_dir, "stage": cfg.data.stage, "view": cfg.data.stage2_view,
                     "steps_available": self.data.n_steps, "doc_mask": cfg.data.doc_mask,
                     "max_rows_per_step": cfg.data.max_rows_per_step, "holdout_groups":
                         {k: len(v) for k, v in self.holdout.by_group.items()}},
            "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(self.dev) if self.dev.type == "cuda" else "cpu",
                    "python": sys.version.split()[0], "platform": platform.platform(),
                    "host": socket.gethostname()},
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_json_atomic(self.run_dir / "meta.json", meta)
        save_config(self.cfg, self.run_dir / "config.yaml")
        if self.log.wandb is not None:
            self.log.wandb.update_config({"meta": {k: meta[k] for k in (
                "params", "runtime", "world_size", "grad_accum", "rows_per_step", "tokens_per_step",
                "stage_start_step", "env")}})
            self.log.wandb.save_files([self.run_dir / "meta.json", self.run_dir / "config.yaml"], self.run_dir)
        p = meta["params"]
        print(f"[lmarch] {cfg.model.arch} {stage_tag(cfg)} pattern={self.raw.pattern} params={p['total'] / 1e6:.1f}M "
              f"(non-emb {p['non_embedding'] / 1e6:.1f}M) world={self.d.world_size} rows/step={self.rows_per_step}"
              f"x{cfg.train.seq_len} micro={cfg.train.micro_batch_size} accum={self.accum} "
              f"steps {self.state.step}->{cfg.train.max_steps} {self.runtime_info}", flush=True)
        if steps_left > self.data.n_steps:
            self.log.event("warning", message=f"stage needs {steps_left} steps but the data has "
                                              f"{self.data.n_steps}: the stream will wrap around")

    def _start_prefetch(self) -> None:
        if self.prefetch is not None:
            self.prefetch.close()
        self.prefetch = Prefetcher(self.data, self.data_index(self.state.step), self.cfg.data.prefetch,
                                   pin=self.dev.type == "cuda")

    def _heartbeat(self, force: bool = False) -> None:
        if not self.d.is_main:
            return
        now = time.time()
        if force or now - self.last_heartbeat >= self.cfg.log.heartbeat_interval_sec:
            self.last_heartbeat = now
            _write_json_atomic(self.run_dir / "heartbeat.json",
                               {"step": self.state.step, "time": now, "pid": os.getpid()})

    # ==================================================================================
    # main loop
    # ==================================================================================
    def run(self) -> int:
        code, reason, detail = EXIT_OK, "completed", ""
        try:
            self.train()
        except StopRequested as e:
            code, reason = e.code, ("preempted" if e.code == EXIT_PREEMPTED else "user_stop")
        except DivergenceError as e:
            code, reason, detail = EXIT_DIVERGED, "diverged", str(e)
            self.log.event("diverged", step=self.state.step, message=str(e))
        except torch.cuda.OutOfMemoryError as e:
            code, reason, detail = EXIT_OOM, "oom", str(e)[:1000]
            self.log.event("oom", step=self.state.step, micro_batch_size=self.cfg.train.micro_batch_size,
                           message=detail[:500])
        except KeyboardInterrupt:
            code, reason = EXIT_USER_STOP, "keyboard_interrupt"
            self._emergency_save()
        except Exception as e:  # noqa: BLE001 - anything else: record, try to save, let supervisor restart
            code, reason, detail = EXIT_ERROR, type(e).__name__, traceback.format_exc()
            self.log.event("error", step=self.state.step, error=repr(e), traceback=detail)
            self._emergency_save()
        finally:
            if self.prefetch is not None:
                self.prefetch.close()
            self._heartbeat(force=True)
            _write_json_atomic(self.run_dir / f"exit_status_rank{self.d.rank}.json",
                               {"code": code, "reason": reason, "detail": redact(detail[:4000]), "step": self.state.step,
                                "micro_batch_size": self.cfg.train.micro_batch_size, "time": time.time()})
            self.log.event("exit", code=code, reason=reason, step=self.state.step)
            self.log.sync()
            self.log.close(code)
        return code

    def train(self) -> None:
        cfg = self.cfg
        self._start_prefetch()
        self.log.event("train_begin", start_step=self.state.step, max_steps=cfg.train.max_steps,
                       data_index=self.data_index(self.state.step), world_size=self.d.world_size,
                       micro_batch_size=cfg.train.micro_batch_size, grad_accum=self.accum)
        while self.state.step < cfg.train.max_steps:
            stop = self._train_step(self.state.step)
            if stop:
                if self.last_ckpt_step != self.state.step:
                    self._save()
                raise StopRequested(stop)
        if cfg.eval.final_full:
            self._evaluate(self.state.step, self._sparse(self.state.step - 1), full=True)
        if cfg.checkpoint.save_final and not (self.ckpt.dir / f"step_{self.state.step:07d}_final").exists():
            self._save(tag="final")
        self.log.event("train_end", step=self.state.step, tokens_seen=self.state.tokens_seen,
                       n_rollbacks=self.state.n_rollbacks, n_skipped=self.state.n_skipped)

    def _sparse(self, step: int) -> bool:
        return step >= self.state.indexer_start_step + self.cfg.model.indexer.dense_warmup_steps

    def _train_step(self, step: int) -> int:
        cfg, st, d = self.cfg, self.state, self.d
        t0 = time.perf_counter()
        lr_frac = lr_multiplier(step, cfg.train.max_steps, cfg.schedule)
        lr = cfg.optim.lr * lr_frac * st.lr_scale
        for g in self.opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        sparse = self._sparse(step)
        flags = RunFlags(sparse=sparse, indexer_loss=self.has_indexer, diag=False)
        diag_step = cfg.diag.interval > 0 and (step % cfg.diag.interval == 0 or step == cfg.train.max_steps - 1)
        if self.dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.dev)
        if step == cfg.debug.inject_exception_step:
            marker = self.run_dir / f".injected_exception_{step}"   # fire once, so a restart can recover
            if not marker.exists():
                marker.touch()
                raise RuntimeError("injected exception (debug.inject_exception_step)")

        di = self.data_index(step)
        td = time.perf_counter()
        item = self.prefetch.get(di)
        data_time = time.perf_counter() - td
        # the loss is averaged over the non-ignored labels of the GLOBAL batch: every rank knows that count,
        # and DDP averages gradients over ranks, hence the factor world_size
        norm = d.world_size / max(item["global_valid"], 1)
        self.model.train()
        loss_sum = torch.zeros((), device=self.dev, dtype=torch.float64)
        valid_sum = torch.zeros((), device=self.dev, dtype=torch.float64)
        aux_sum = torch.zeros((), device=self.dev)
        grp_loss = torch.zeros(N_GROUPS, device=self.dev, dtype=torch.float64)
        grp_cnt = torch.zeros(N_GROUPS, device=self.dev, dtype=torch.float64)
        kl_sum = None
        micro_losses, micro_batches = [], []
        for micro, mb in enumerate(item["micro"]):
            x, y, pos = _to_dev(mb["ids"], self.dev), _to_dev(mb["labels"], self.dev), _to_dev(mb["pos"], self.dev)
            grp = mb["group"].to(self.dev, non_blocking=True)
            micro_batches.append((x, y, pos))
            sync = self.model.no_sync() if (d.is_dist and micro < self.accum - 1) else contextlib.nullcontext()
            with sync:
                with self._autocast():
                    out = self.model(x, y, flags, pos)
                ls = out["loss_sum"]
                if di in cfg.debug.inject_nan_steps:      # simulated poisoned batch
                    ls = ls * float("nan")
                if di in cfg.debug.inject_spike_steps:
                    ls = ls * 10.0
                total = ls * norm
                if "aux_loss" in out:
                    total = total + out["aux_loss"] / self.accum
                if "z_sum" in out:
                    total = total + cfg.train.z_loss_coef * out["z_sum"] * norm
                total.backward()
            loss_sum += ls.detach().double()
            valid_sum += out["n_valid"].double()
            micro_losses.append(ls.detach().float() / out["n_valid"].clamp(min=1).float())
            grp_loss.index_add_(0, grp, out["row_loss"].double())
            grp_cnt.index_add_(0, grp, out["row_valid"].double())
            if "aux_loss" in out:
                aux_sum += out["aux_loss"].detach().float()
            if "indexer_kl" in out:
                kl_sum = out["indexer_kl"] if kl_sum is None else kl_sum + out["indexer_kl"]

        # ---- cross-rank reduction (one SUM + one MAX all-reduce) -------------------------
        kl_vec = (kl_sum / self.accum).double() if kl_sum is not None else torch.zeros(0, device=self.dev, dtype=torch.float64)
        vec = torch.cat([torch.stack([loss_sum, valid_sum, aux_sum.double() / self.accum]), kl_vec, grp_loss, grp_cnt])
        all_reduce_(vec, d, "sum")
        flg = torch.tensor([float(not torch.isfinite(loss_sum).item()), float(self.stop_code)], device=self.dev)
        all_reduce_(flg, d, "max")
        clip = cfg.train.grad_clip if cfg.train.grad_clip > 0 else float("inf")
        gnorm_t = torch.nn.utils.clip_grad_norm_(self.raw.parameters(), clip)
        vals = torch.cat([vec, flg.double(), gnorm_t.reshape(1).double()]).cpu().tolist()
        nk = kl_vec.numel()
        g_loss, g_cnt = vals[3 + nk:3 + nk + N_GROUPS], vals[3 + nk + N_GROUPS:3 + nk + 2 * N_GROUPS]
        lm_loss = vals[0] / max(vals[1], 1.0)
        aux_loss = vals[2] / d.world_size
        kl_layers = [v / d.world_size for v in vals[3:3 + nk]]
        local_bad, stop_code, gnorm = vals[-3], int(vals[-2]), vals[-1]
        loss_finite = math.isfinite(lm_loss) and math.isfinite(aux_loss) and local_bad == 0
        grad_finite = math.isfinite(gnorm)

        grad_stats, snap = None, None
        if diag_step and d.is_main and grad_finite:
            grad_stats = self.pindex.grad_norms()
            snap = self.pindex.snapshot()
        if not (loss_finite and grad_finite):
            self._forensics(step, micro_batches, micro_losses, flags)
        verdict = self.guard.check(step, lm_loss, gnorm, loss_finite, grad_finite)
        applied = verdict.action == "ok"
        if applied:
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        upd_stats = self.pindex.update_stats(snap) if (snap is not None and applied) else None
        del snap
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        step_time = time.perf_counter() - t0
        mem = torch.zeros(2, device=self.dev)
        if self.dev.type == "cuda":
            mem[0] = torch.cuda.max_memory_allocated(self.dev) / 2**30
            mem[1] = torch.cuda.max_memory_reserved(self.dev) / 2**30
        all_reduce_(mem, d, "max")
        peak_gb, reserved_gb = mem.tolist()

        by_src, by_rt = {}, {}
        for gi in range(N_GROUPS):
            if g_cnt[gi] > 0:
                s_name, r_name = group_name(gi).split("/")
                a = by_src.setdefault(s_name, [0.0, 0.0]); a[0] += g_loss[gi]; a[1] += g_cnt[gi]
                b = by_rt.setdefault(r_name, [0.0, 0.0]); b[0] += g_loss[gi]; b[1] += g_cnt[gi]
        kl_by_type = {}
        for t, v in zip(self.indexer_types, kl_layers):
            kl_by_type.setdefault(t, []).append(v)
        rec = {
            "step": step, "time": time.time(), "tokens_seen": st.tokens_seen + self.tokens_per_step,
            "data_index": di, "data_batch": item["batch"], "data_epoch": item["epoch"],
            "lr": lr, "lr_frac": lr_frac, "lr_scale": st.lr_scale,
            "loss": lm_loss, "valid_tokens": int(vals[1]), "aux_loss": aux_loss,
            "loss_by_source": {k: v[0] / v[1] for k, v in by_src.items()},
            "loss_by_row_type": {k: v[0] / v[1] for k, v in by_rt.items()},
            "long_token_frac": item["long_frac"], "mean_segment_len": item["mean_segment_len"],
            "loss_nonfinite": not loss_finite, "grad_nonfinite": not grad_finite,
            "grad_norm": gnorm, "grad_clipped": bool(grad_finite and gnorm > clip),
            "skipped": not applied, "action": verdict.action, "spike": verdict.spike, "reasons": verdict.reasons,
            "step_time": step_time, "data_time": data_time,
            "tokens_per_sec": self.tokens_per_step / step_time,
            "tokens_per_sec_per_gpu": self.tokens_per_step / step_time / d.world_size,
            "peak_mem_gb": peak_gb, "peak_mem_reserved_gb": reserved_gb,
            "sparse_phase": sparse if self.has_indexer else None, "loss_ema": self.guard.ema,
        }
        if kl_layers:
            rec["indexer_kl"] = sum(kl_layers) / len(kl_layers)
            rec["indexer_kl_by_type"] = {t: sum(v) / len(v) for t, v in kl_by_type.items()}
            rec["indexer_kl_by_layer"] = kl_layers
        self.log.step(rec)
        if verdict.reasons:
            self.log.event("guard", step=step, action=verdict.action, reasons=verdict.reasons,
                           loss=lm_loss, grad_norm=gnorm)
        if verdict.action == "rollback":
            self._rollback(step, verdict)
            return stop_code

        st.step += 1
        st.tokens_seen += self.tokens_per_step
        if not applied:
            st.n_skipped += 1
        if diag_step and d.is_main:
            self._diagnostics(step, lr_frac * st.lr_scale, grad_stats, upd_stats, sparse)
        if d.is_main and (step % cfg.log.console_interval == 0 or verdict.reasons):
            self._console(rec)
        if cfg.eval.interval > 0 and st.step % cfg.eval.interval == 0 and st.step < cfg.train.max_steps:
            self._evaluate(st.step, sparse)
        if cfg.checkpoint.interval > 0 and st.step % cfg.checkpoint.interval == 0 and st.step < cfg.train.max_steps:
            self._save()
        self._heartbeat()
        return stop_code

    def _console(self, r: dict) -> None:
        msg = (f"step {r['step']:>6}/{self.cfg.train.max_steps} | loss {r['loss']:.4f} | lr {r['lr']:.2e} | "
               f"gnorm {r['grad_norm']:.3f} | {r['step_time']:.2f}s | {r['tokens_per_sec'] / 1e3:.1f}k tok/s | "
               f"mem {r['peak_mem_gb']:.1f}GB")
        if "indexer_kl" in r:
            msg += f" | idxKL {r['indexer_kl']:.3f}" + ("" if r["sparse_phase"] else " (dense warm-up)")
        if r["action"] != "ok" or r["reasons"]:
            msg += f" | {r['action'].upper()} {'; '.join(r['reasons'])}"
        print(msg, flush=True)

    # ==================================================================================
    # diagnostics / eval
    # ==================================================================================
    @torch.no_grad()
    def _logit_stats(self, hidden: torch.Tensor, chunk: int = 1024) -> dict:
        """max / std / mean logsumexp of the LM logits, computed in chunks (128K vocab)."""
        w = self.raw.lm_head.weight
        h = hidden.reshape(-1, hidden.shape[-1])
        mx, mn, s1, s2, lse, n = -float("inf"), float("inf"), 0.0, 0.0, 0.0, 0
        for i in range(0, h.shape[0], chunk):
            lg = torch.nn.functional.linear(h[i:i + chunk], w).float()
            mx, mn = max(mx, lg.max().item()), min(mn, lg.min().item())
            s1 += lg.sum().item()
            s2 += lg.pow(2).sum().item()
            lse += torch.logsumexp(lg, -1).sum().item()
            n += lg.shape[0]
        numel = n * w.shape[0]
        return {"logit_max": mx, "logit_min": mn,
                "logit_std": math.sqrt(max(s2 / numel - (s1 / numel) ** 2, 0.0)), "logit_lse_mean": lse / n}

    @torch.no_grad()
    def _diagnostics(self, step: int, lr_frac: float, grad_stats, upd_stats, sparse: bool) -> None:
        t0 = time.perf_counter()
        raw = self.raw
        raw.eval()
        p = self.probe
        x = torch.as_tensor(p["ids"], device=self.dev)
        y = torch.as_tensor(p["labels"], device=self.dev)
        pos = torch.as_tensor(p["pos"], device=self.dev) if p["pos"] is not None else None
        with self._autocast():
            out = raw(x, y, RunFlags(sparse=sparse, indexer_loss=False, diag=True), pos, return_hidden=True)
        layers = collect_probe(raw)
        glob = {"probe_loss": (out["loss_sum"] / out["n_valid"].clamp(min=1)).item(),
                "probe_rows": p["names"]}
        glob.update(self._logit_stats(out["hidden"]))
        del out
        raw.train()
        grad_stats, upd_stats = grad_stats or {}, upd_stats or {}
        for L in layers:
            key = f"L{L['idx']:02d}"
            gn = {g: grad_stats[(key, g)] for g in GROUPS if (key, g) in grad_stats}
            if gn:
                gn["total"] = math.sqrt(sum(v * v for v in gn.values()))
            L["grad_norm"] = gn
            L["update"] = {g: dict(upd_stats[(key, g)], health=health(upd_stats[(key, g)]["ratio"], self.cfg.diag))
                           for g in GROUPS if (key, g) in upd_stats}
        for key in ("embed", "final_norm"):
            grp = "embed" if key == "embed" else "norm"
            if (key, grp) in grad_stats:
                glob[f"{key}_grad_norm"] = grad_stats[(key, grp)]
            if (key, grp) in upd_stats:
                glob[f"{key}_update_ratio"] = upd_stats[(key, grp)]["ratio"]
                glob[f"{key}_weight_norm"] = upd_stats[(key, grp)]["w_norm"]
        if upd_stats:
            glob["param_norm_total"] = math.sqrt(sum(v["w_norm"] ** 2 for v in upd_stats.values()))
        alerts = self.alerts.update(upd_stats, lr_frac) if upd_stats else []
        for a in alerts:
            self.log.event("ratio_alert", step=step, **a)
        self.log.diag({"step": step, "lr_frac": lr_frac, "sparse_phase": sparse, "global": glob,
                       "by_type": group_by_type(layers), "layers": layers, "alerts": alerts,
                       "diag_time": time.perf_counter() - t0})

    @torch.no_grad()
    def _evaluate(self, step: int, sparse: bool, full: bool = False) -> None:
        """Holdout loss at train.seq_len (+ eval.extra_seq_lens): overall, per source x row_type group and
        per position-in-document bucket; for DSA/CSA also with dense attention (cost of sparsity)."""
        cfg, d, raw = self.cfg, self.d, self.raw
        t0 = time.perf_counter()
        rows_per_group = 0 if full else cfg.eval.rows_per_group
        lens = [cfg.train.seq_len] + [int(L) for L in cfg.eval.extra_seq_lens if int(L) != cfg.train.seq_len]
        raw.eval()
        rec: dict = {"step": step, "tokens_seen": self.state.tokens_seen, "full": full,
                     "rows_per_group": rows_per_group}
        for L in lens:
            edges = [0] + sorted(b for b in cfg.eval.position_buckets if 0 < b < L) + [L]
            modes = [("main", sparse)]
            if self.has_indexer and cfg.eval.dense_compare and sparse and L == cfg.train.seq_len:
                modes.append(("dense", False))
            micro = self.eval_micro if L <= cfg.train.seq_len else max(1, self.eval_micro * cfg.train.seq_len // L)
            batches = self.holdout.micro_batches(rows_per_group, L, d.rank, d.world_size, micro, cfg.data.doc_mask)
            sfx = "" if L == cfg.train.seq_len else f"@{L}"
            for name, sp in modes:
                flags = RunFlags(sparse=sp, indexer_loss=False, diag=False)
                g_sum = torch.zeros(N_GROUPS, device=self.dev, dtype=torch.float64)
                g_cnt = torch.zeros(N_GROUPS, device=self.dev, dtype=torch.float64)
                b_sum = torch.zeros(len(edges) - 1, device=self.dev, dtype=torch.float64)
                b_cnt = torch.zeros(len(edges) - 1, device=self.dev, dtype=torch.float64)
                for mb in batches:
                    x = torch.as_tensor(mb["ids"], device=self.dev)
                    y = torch.as_tensor(mb["labels"], device=self.dev)
                    pos = torch.as_tensor(mb["pos"], device=self.dev) if mb["pos"] is not None else None
                    grp = torch.as_tensor(mb["group"], device=self.dev)
                    with self._autocast():
                        out = raw(x, y, flags, pos, per_token=True)
                    g_sum.index_add_(0, grp, out["row_loss"].double())
                    g_cnt.index_add_(0, grp, out["row_valid"].double())
                    tok, valid = out["tok_loss"].double(), (y != -100)
                    where = pos if pos is not None else torch.arange(L, device=self.dev).expand_as(y)
                    bucket = torch.bucketize(where, torch.tensor(edges[1:-1], device=self.dev), right=True)
                    b_sum.index_add_(0, bucket[valid], tok[valid])
                    b_cnt.index_add_(0, bucket[valid], torch.ones_like(tok[valid]))
                red = torch.cat([g_sum, g_cnt, b_sum, b_cnt])
                all_reduce_(red, d, "sum")
                g_sum, g_cnt = red[:N_GROUPS], red[N_GROUPS:2 * N_GROUPS]
                b_sum, b_cnt = red[2 * N_GROUPS:2 * N_GROUPS + len(edges) - 1], red[2 * N_GROUPS + len(edges) - 1:]
                loss = (g_sum.sum() / g_cnt.sum().clamp(min=1)).item()
                if name == "main":
                    rec[f"val_loss{sfx}"] = loss
                    rec[f"val_ppl{sfx}"] = math.exp(min(loss, 50.0))
                    rec[f"val_by_group{sfx}"] = {group_name(i): (g_sum[i] / g_cnt[i]).item()
                                                  for i in range(N_GROUPS) if g_cnt[i] > 0}
                    rec[f"val_by_pos{sfx}"] = {f"{edges[i]}-{edges[i + 1]}": (b_sum[i] / b_cnt[i]).item()
                                                for i in range(len(edges) - 1) if b_cnt[i] > 0}
                else:
                    rec[f"val_loss_dense_attn{sfx}"] = loss
            if f"val_loss_dense_attn{sfx}" in rec:
                rec[f"sparse_minus_dense{sfx}"] = rec[f"val_loss{sfx}"] - rec[f"val_loss_dense_attn{sfx}"]
        raw.train()
        rec["eval_time"] = time.perf_counter() - t0
        if not full and (self.state.best_val is None or rec["val_loss"] < self.state.best_val):
            self.state.best_val = rec["val_loss"]
        self.log.eval(rec)
        if d.is_main:
            extra = "".join(f" | val@{L} {rec[f'val_loss@{L}']:.4f}" for L in lens[1:])
            dense = f" | dense-attn {rec['val_loss_dense_attn']:.4f}" if "val_loss_dense_attn" in rec else ""
            groups = ", ".join(f"{k}:{v:.3f}" for k, v in rec["val_by_group"].items())
            print(f"[eval{' FULL' if full else ''}] step {step} | val_loss {rec['val_loss']:.4f}{dense}{extra} "
                  f"| {groups} ({rec['eval_time']:.0f}s)", flush=True)

    # ==================================================================================
    # checkpointing / recovery
    # ==================================================================================
    def _save(self, tag: str = "") -> None:
        healthy = self.guard.is_healthy(self.state.step)
        path = self.ckpt.save(self.state.step, self.raw, self.opt, self._state_payload(healthy), tag)
        self.last_ckpt_step = self.state.step
        self.log.event("checkpoint", step=self.state.step, path=str(path), healthy=healthy,
                       seconds=round(self.ckpt.last_save_seconds, 2))
        self.log.sync()

    def _emergency_save(self) -> None:
        """Best effort after an exception. Single-process only: with DDP the other ranks may be stuck in
        a collective, so we rely on the periodic checkpoints instead."""
        if self.d.is_dist or self.state.step <= self.last_ckpt_step:
            return
        try:
            if not all(torch.isfinite(p).all() for p in self.raw.parameters()):
                return
            self._save(tag="emergency")
        except Exception as e:  # noqa: BLE001
            print(f"[emergency save failed] {e!r}", flush=True)

    def _rollback(self, fail_step: int, verdict) -> None:
        g = self.cfg.guard
        prev = asdict(self.state)
        n_rb = prev["n_rollbacks"] + 1
        if n_rb > g.max_rollbacks:
            raise DivergenceError(f"divergence at step {fail_step} after {n_rb - 1} rollbacks "
                                  f"({'; '.join(verdict.reasons)})")
        cands = [p for p in self.ckpt.candidates(healthy_only=True, max_step=fail_step)
                 if int(p.name.split("_")[1]) >= prev["stage_start_step"]]
        if not cands:
            raise DivergenceError(f"no healthy checkpoint of this stage at or before step {fail_step}")
        # a checkpoint that keeps leading back into divergence may already be "sick": go further back
        target = next((p for p in cands if self.rollback_counts[str(p)] < g.max_rollbacks_per_checkpoint),
                      cands[-1])
        self.rollback_counts[str(target)] += 1
        if self.prefetch is not None:
            self.prefetch.close()
        loaded = self.ckpt.load(target, self.raw, self.opt, restore_rng=True)
        self._apply_loaded_state(loaded)
        c = self.state.step
        skipped = fail_step + 1 + g.rollback_skip_extra - c
        # data index of step c becomes the one that originally followed the failure (+ extra)
        self.state.data_skip = prev["data_skip"] + skipped
        self.state.lr_scale = prev["lr_scale"] * g.lr_scale_on_rollback
        self.state.n_rollbacks = n_rb
        self.state.n_skipped = prev["n_skipped"]
        self.state.segment = max(prev["segment"], self.state.segment) + 1
        self.log.segment = self.state.segment
        self.last_ckpt_step = c
        self.log.event("rollback", from_step=fail_step, to_step=c, checkpoint=str(target),
                       skipped_batches=skipped, data_skip=self.state.data_skip, next_data_index=self.data_index(c),
                       lr_scale=self.state.lr_scale, n_rollbacks=n_rb, reasons=verdict.reasons)
        barrier(self.d)
        self._start_prefetch()

    def _forensics(self, step: int, micro_batches, micro_losses, flags: RunFlags) -> None:
        """Locate where non-finite values come from: parameters with bad grads (grouped by layer type)
        and the first module whose forward output is non-finite when the batch is replayed."""
        if not self.cfg.guard.forensics or self.forensics_done >= self.cfg.guard.max_forensics:
            return
        self.forensics_done += 1
        rep: dict = {"step": step, "rank": self.d.rank, "data_index": self.data_index(step), "time": time.time()}
        ml = torch.stack(micro_losses).cpu().tolist()
        rep["micro_losses"] = ml
        types = self.pindex.layer_types
        if self.d.is_main:
            bad_g, bad_w, by_type = [], [], Counter()
            for name, p, key, grp in self.pindex.entries:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_g.append(name)
                    by_type[f"{types.get(key, key)}/{grp}"] += 1
                if not torch.isfinite(p).all():
                    bad_w.append(name)
            rep["nonfinite_grad_params"] = bad_g[:200]
            rep["nonfinite_grad_by_type"] = dict(by_type)
            rep["nonfinite_weights"] = bad_w[:200]
        bad_micro = [i for i, v in enumerate(ml) if not math.isfinite(v)]
        if bad_micro or self.d.is_main:
            i = bad_micro[0] if bad_micro else len(micro_batches) - 1
            x, y, pos = micro_batches[i]
            rep["replay_micro"] = i
            rep.update(self._first_nonfinite_module(x, y, pos, flags))
            torch.save({"x": x.cpu(), "y": y.cpu(), "pos": None if pos is None else pos.cpu(), "step": step,
                        "micro": i, "data_index": self.data_index(step)},
                       self.run_dir / "crash_reports" / f"step{step:07d}_rank{self.d.rank}_batch.pt")
        _write_json_atomic(self.run_dir / "crash_reports" / f"step{step:07d}_rank{self.d.rank}.json", rep)
        self.log.event("nonfinite_forensics", step=step, first_nonfinite=rep.get("first_nonfinite_module"),
                       layer_type=rep.get("first_nonfinite_layer_type"),
                       grad_by_type=rep.get("nonfinite_grad_by_type"))

    @torch.no_grad()
    def _first_nonfinite_module(self, x, y, pos, flags: RunFlags) -> dict:
        names = {m: n for n, m in self.raw.named_modules()}
        found: list = []

        def hook(mod, inp, out):
            if found:
                return
            outs = out if isinstance(out, (tuple, list)) else (out,)
            for o in outs:
                if torch.is_tensor(o) and o.is_floating_point() and not torch.isfinite(o).all():
                    found.append(names.get(mod, "?"))
                    return

        handles = [m.register_forward_hook(hook) for m in self.raw.modules()]
        try:
            with self._autocast():
                out = self.raw(x, y, RunFlags(sparse=flags.sparse, indexer_loss=False, diag=False), pos)
            replay_loss = (out["loss_sum"] / out["n_valid"].clamp(min=1)).item()
        finally:
            for h in handles:
                h.remove()
        res = {"replay_loss": replay_loss, "first_nonfinite_module": found[0] if found else None}
        if found and found[0].startswith("blocks."):
            i = int(found[0].split(".")[1])
            res["first_nonfinite_layer"] = i
            res["first_nonfinite_layer_type"] = LAYER_TYPE_NAMES[self.raw.pattern[i]]
        elif not found:
            res["note"] = "forward is finite on replay: non-finite arose in the loss/backward or is data-order dependent"
        return res


def _dict_diff(a: dict, b: dict, prefix: str = "") -> dict:
    out = {}
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(_dict_diff(va, vb, f"{prefix}{k}."))
        elif va != vb:
            out[f"{prefix}{k}"] = {"checkpoint": va, "now": vb}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train one architecture of the comparison.")
    ap.add_argument("--config", required=True, help="YAML config (see configs/)")
    ap.add_argument("--arch", choices=sorted(ARCH_PRESETS), help="dense | dsa | kda_full | kda_dsa | csa")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--resume", default=None, help="auto | none | <checkpoint dir>")
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE",
                    help="config overrides, e.g. --set optim.lr=3e-4 train.max_steps=100 (repeatable)")
    args = ap.parse_args(argv)
    # API keys (W&B, HF, GitHub) from the secrets file into this process's environment; names only are printed
    load_secrets(verbose=os.environ.get("RANK", "0") == "0")
    overrides = list(args.set)
    if args.resume:
        overrides.append(f"checkpoint.resume={args.resume}")
    try:
        cfg = build_config(args.config, args.arch, args.run_name, overrides)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_CONFIG
    d = setup_distributed(cfg.train.nccl_timeout_min)
    try:
        trainer = Trainer(cfg, d)
    except (ConfigError, FileNotFoundError, ValueError, ImportError) as e:
        print(f"setup error: {e}", file=sys.stderr)
        run_dir = Path(cfg.log.out_dir) / cfg.log.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(run_dir / f"exit_status_rank{d.rank}.json",
                           {"code": EXIT_CONFIG, "reason": "setup_error", "detail": str(e), "time": time.time()})
        cleanup(d)
        return EXIT_CONFIG
    code = trainer.run()
    cleanup(d)
    return code


if __name__ == "__main__":
    sys.exit(main())
