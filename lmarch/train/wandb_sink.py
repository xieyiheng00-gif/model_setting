"""Weights & Biases mirror of the run logs (rank 0 only). The JSONL files stay the source of truth;
nothing in here may ever stop training: every W&B call is wrapped and W&B is switched off on failure.

API key: never put it in a config or the repo. On each machine run `wandb login` once (stored in
~/.netrc, or _netrc on Windows) or export WANDB_API_KEY (e.g. as a cluster secret). Without a key the
run falls back to offline mode; upload later with `wandb sync <run_dir>/wandb/offline-run-*`.

Layout in the W&B UI (one section per prefix):
  train/*            every step: loss, lr, grad_norm, NaN/Inf flags, skipped, step_time, tokens/sec, memory, ...
  train_source/*     per-step loss by data source;  train_rowtype/* by row type (long/short/trunk)
  diag_<type>/*      every diag.interval steps, per layer TYPE (softmax/dsa/kda/csa): mean and .max over its layers
  layer_<metric>/*   the same metric for every layer, e.g. layer_update_mixer/L03_kda (one panel per layer)
  diag/*             probe loss, LM-logit stats, embedding stats, number of ratio alerts
  eval/*, eval_group/*, eval_pos/*   holdout loss (periodic); eval_full*/* for the full holdout at stage end
  events/*           cumulative counts: rollbacks, NaN forensics, ratio alerts, resumes, checkpoints
x-axis: trainer/step (the optimizer step). After a rollback or a crash-resume the re-run steps are logged
again (train/segment increments), so curves can briefly overlap; the JSONL analysis keeps the last one.
W&B alerts (email/Slack, per your W&B settings) fire on rollbacks, divergence, OOM, crashes, persistent
update-ratio alerts and at the end of a stage.
"""
from __future__ import annotations

import math
import netrc
import os
from pathlib import Path
from urllib.parse import urlparse

LAYER_METRICS = {                       # per-layer series (key inside the layer record -> panel name)
    ("act", "act_rms"): "act_rms",
    ("act", "act_absmax"): "act_absmax",
    ("update", "mixer", "ratio"): "update_mixer",
    ("update", "ffn", "ratio"): "update_ffn",
    ("update", "indexer", "ratio"): "update_indexer",
    ("grad_norm", "total"): "grad_norm",
    ("mixer", "max_logit"): "max_logit",
    ("mixer", "recall_mean"): "recall",
    ("mixer", "forget_gate", "mean"): "forget_gate_mean",
    ("mixer", "beta", "mean"): "beta_mean",
    ("mixer", "state_norm_max"): "kda_state_norm",
    ("mixer", "sink_mass"): "sink_mass",
}
ALERT_EVENTS = {"rollback": "WARN", "diverged": "ERROR", "oom": "ERROR", "error": "ERROR",
                "ratio_alert": "WARN", "checkpoint_corrupt": "WARN"}
COUNT_EVENTS = ("rollback", "nonfinite_forensics", "ratio_alert", "resume", "checkpoint", "guard",
                "checkpoint_corrupt")


def has_api_key() -> bool:
    """True if a W&B key is available to this process (env var or netrc); the key itself is never read out."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname or "api.wandb.ai"
    home = Path.home()
    for f in (os.environ.get("NETRC"), home / ".netrc", home / "_netrc"):
        if f and Path(f).exists():
            try:
                if netrc.netrc(str(f)).authenticators(host):
                    return True
            except (netrc.NetrcParseError, OSError):
                continue
    return False


def _num(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return v
    return None


def _flat(d: dict, prefix: str, out: dict, sep: str = "/") -> dict:
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flat(v, key, out, sep)
        else:
            n = _num(v)
            if n is not None:
                out[key] = n
    return out


def _dig(d, path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    if isinstance(d, list) and d and all(isinstance(x, (int, float)) for x in d):
        return max(d)                         # per-head lists -> worst head
    return _num(d)


class WandbSink:
    def __init__(self, lcfg, run_dir: Path, run_name: str, config: dict, tags: list[str], group: str,
                 job_type: str):
        self.run = None
        self.alerts = lcfg.wandb_alerts
        self.log_layers = lcfg.wandb_log_layers
        self.run_name = run_name
        self.counts: dict = {}
        self.last_step = 0
        self.status = "disabled"            # disabled | online | offline | failed
        self.error: str | None = None
        mode = lcfg.wandb_mode
        if mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            self.status, self.error = "failed", "wandb package not installed (pip install wandb)"
            print(f"[wandb] WARNING: {self.error}: W&B logging is OFF for this run", flush=True)
            return
        if mode == "online" and not has_api_key():
            print("[wandb] no API key on this machine (run `wandb login` or set WANDB_API_KEY): logging OFFLINE "
                  f"to {run_dir / 'wandb'}; upload later with `wandb sync {run_dir / 'wandb'}/offline-run-*`",
                  flush=True)
            mode = "offline"
        id_file = run_dir / "wandb_run_id.txt"      # the same W&B run survives crashes / restarts
        try:
            if id_file.exists():
                run_id = id_file.read_text(encoding="utf-8").strip()
            else:  # wandb.util.generate_id was moved to wandb.sdk.lib.runid in newer wandb releases
                try:
                    from wandb.sdk.lib.runid import generate_id
                except ImportError:
                    generate_id = wandb.util.generate_id
                run_id = generate_id()
            id_file.write_text(run_id, encoding="utf-8")
            self.run = wandb.init(project=lcfg.wandb_project, entity=lcfg.wandb_entity or None, name=run_name,
                                  id=run_id, resume="allow", group=group or None, job_type=job_type,
                                  tags=tags, config=config, dir=str(run_dir), mode=mode,
                                  settings=wandb.Settings(init_timeout=180))
            self.run.define_metric("trainer/step")
            self.run.define_metric("*", step_metric="trainer/step")
            for key, summ in (("train/loss", "min"), ("eval/val_loss", "min"), ("eval_full/val_loss", "last"),
                              ("train/tokens_per_sec", "mean"), ("train/peak_mem_gb", "max"),
                              ("train/grad_norm", "max")):
                self.run.define_metric(key, summary=summ)
            self._wandb = wandb
            self.status = mode
            url = getattr(self.run, "url", None)
            print(f"[wandb] {mode}: {url or run_dir / 'wandb'}", flush=True)
        except Exception as e:  # noqa: BLE001 - never block training on W&B
            # recorded as a `wandb` event with status=failed (it used to be one easily-missed console line)
            self.status, self.error = "failed", f"init failed: {type(e).__name__}: {e}"
            print(f"[wandb] WARNING: {self.error}: W&B logging is OFF for this run "
                  f"(training continues; metrics are still written to {run_dir / 'metrics'})", flush=True)
            self.run = None

    # ---- helpers ------------------------------------------------------------------------
    def _log(self, payload: dict, step: int) -> None:
        if self.run is None or not payload:
            return
        try:
            payload["trainer/step"] = step
            self.run.log(payload)
            self.last_step = max(self.last_step, step)
        except Exception as e:  # noqa: BLE001
            self._disable(e)

    def _disable(self, e: Exception) -> None:
        self.status, self.error = "failed", f"logging failed: {type(e).__name__}: {e}"
        print(f"[wandb] logging failed ({type(e).__name__}: {e}): W&B disabled for the rest of the run", flush=True)
        try:
            self.run.finish(exit_code=1)
        except Exception:
            pass
        self.run = None

    # ---- records ------------------------------------------------------------------------
    def train(self, rec: dict) -> None:
        if self.run is None:
            return
        p: dict = {}
        for k, v in rec.items():
            if k in ("step", "time", "reasons", "indexer_kl_by_layer"):
                continue
            if k == "loss_by_source":
                _flat(v, "train_source", p)
            elif k == "loss_by_row_type":
                _flat(v, "train_rowtype", p)
            elif k == "indexer_kl_by_type":
                _flat({f"indexer_kl_{t}": x for t, x in v.items()}, "train", p)
            elif k == "action":
                p["train/rollback"] = int(v == "rollback")
            else:
                n = _num(v)
                if n is not None:
                    p[f"train/{k}"] = n
        self._log(p, rec["step"])

    def diag(self, rec: dict) -> None:
        if self.run is None:
            return
        p: dict = {}
        for t, metrics in rec.get("by_type", {}).items():
            for m, st in metrics.items():
                p[f"diag_{t}/{m}"] = st["mean"]
                if st.get("n", 1) > 1:
                    p[f"diag_{t}/{m}.max"] = st["max"]
        _flat(rec.get("global", {}), "diag", p)
        p["diag/n_alerts"] = len(rec.get("alerts", []))
        if self.log_layers:
            for L in rec.get("layers", []):
                name = f"L{L['idx']:02d}_{L['type']}"
                for path, panel in LAYER_METRICS.items():
                    v = _dig(L, path)
                    if v is not None:
                        p[f"layer_{panel}/{name}"] = v
        self._log(p, rec["step"])

    def eval(self, rec: dict) -> None:
        if self.run is None:
            return
        pre = "eval_full" if rec.get("full") else "eval"
        p: dict = {}
        for k, v in rec.items():
            if k in ("step", "full", "rows_per_group"):
                continue
            if isinstance(v, dict):                 # val_by_group{@L}, val_by_pos{@L}
                sub = k.replace("val_by_", "")
                for kk, vv in v.items():
                    n = _num(vv)
                    if n is not None:
                        p[f"{pre}_{sub}/{kk.replace('/', '.')}"] = n
            else:
                n = _num(v)
                if n is not None:
                    p[f"{pre}/{k}"] = n
        self._log(p, rec["step"])
        if rec.get("full") and self.run is not None:
            try:
                self.run.summary.update({f"final/{k}": v for k, v in p.items()})
            except Exception:
                pass

    def event(self, kind: str, data: dict) -> None:
        if self.run is None:
            return
        step = data.get("step", data.get("from_step", self.last_step))
        step = step if isinstance(step, int) else self.last_step
        if kind in COUNT_EVENTS:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            self._log({f"events/{kind}": self.counts[kind]}, step)
        if kind == "train_end":
            self._alert("INFO", "stage finished", f"step {step}, rollbacks {data.get('n_rollbacks')}, "
                                                  f"skipped steps {data.get('n_skipped')}")
        elif kind in ALERT_EVENTS:
            detail = {k: v for k, v in data.items() if k not in ("traceback", "diff") and not isinstance(v, (dict, list))}
            text = ", ".join(f"{k}={v}" for k, v in detail.items())
            if "reasons" in data:
                text += " | " + "; ".join(map(str, data["reasons"]))
            self._alert(ALERT_EVENTS[kind], kind, text[:900], wait=0 if kind != "ratio_alert" else 1800)

    def _alert(self, level: str, title: str, text: str, wait: int = 0) -> None:
        if self.run is None or not self.alerts or self.status != "online":
            return
        try:
            self.run.alert(title=f"{self.run_name}: {title}", text=text,
                           level=getattr(self._wandb.AlertLevel, level), wait_duration=wait)
        except Exception:
            pass

    def update_config(self, d: dict) -> None:
        if self.run is not None:
            try:
                self.run.config.update(d, allow_val_change=True)
            except Exception:
                pass

    def save_files(self, paths: list[Path], base: Path) -> None:
        if self.run is not None:
            for p in paths:
                try:
                    self.run.save(str(p), base_path=str(base), policy="now")
                except Exception:
                    pass

    def finish(self, exit_code: int) -> None:
        if self.run is not None:
            try:
                self.run.summary.update({"exit_code": exit_code})
                self.run.finish(exit_code=exit_code)
            except Exception:
                pass
            self.run = None
