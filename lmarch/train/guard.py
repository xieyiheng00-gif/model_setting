"""Divergence guard: decides per step whether to apply the update, skip it, or roll back.

Escalation ladder
  1. non-finite loss/grad            -> skip the optimizer step (grads dropped), log + forensics
  2. >= max_consecutive_bad in a row,
     or >= max_bad_in_window         -> ROLLBACK to the last healthy checkpoint, skip the offending
                                        data window, optionally scale the LR down
  3. loss spike (z-score vs EMA)     -> log (optionally skip); >= spike_rollback_count in
                                        spike_window -> ROLLBACK
  4. > max_rollbacks                 -> stop with exit code 4 (needs a human: lower LR, QK-norm, ...)
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..config import GuardConfig


@dataclass
class Verdict:
    action: str = "ok"                 # ok | skip | rollback
    reasons: list = field(default_factory=list)
    spike: bool = False


class TrainingGuard:
    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg
        self.ema: float | None = None
        self.ema_var = 0.0
        self.n = 0
        self.gn_ema: float | None = None
        self.consecutive_bad = 0
        self.bad_steps: deque = deque()
        self.spike_steps: deque = deque()
        self.last_incident_step = -10**9

    # ---- state --------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"ema": self.ema, "ema_var": self.ema_var, "n": self.n, "gn_ema": self.gn_ema,
                "last_incident_step": self.last_incident_step}

    def load_state_dict(self, d: dict) -> None:
        self.ema, self.ema_var, self.n = d.get("ema"), d.get("ema_var", 0.0), d.get("n", 0)
        self.gn_ema = d.get("gn_ema")
        self.last_incident_step = d.get("last_incident_step", -10**9)
        self.reset_counters()

    def reset_counters(self) -> None:
        self.consecutive_bad = 0
        self.bad_steps.clear()
        self.spike_steps.clear()

    def is_healthy(self, step: int) -> bool:
        """Healthy = no incident in the last `healthy_window` steps -> safe rollback target."""
        return step - self.last_incident_step > self.cfg.healthy_window

    # ---- decision -----------------------------------------------------------------------
    def check(self, step: int, loss: float, grad_norm: float, loss_finite: bool, grad_finite: bool) -> Verdict:
        c = self.cfg
        v = Verdict()
        if not (loss_finite and grad_finite):
            self.last_incident_step = step
            self.consecutive_bad += 1
            self.bad_steps.append(step)
            while self.bad_steps and self.bad_steps[0] <= step - c.bad_window:
                self.bad_steps.popleft()
            if not loss_finite:
                v.reasons.append("nonfinite_loss")
            if not grad_finite:
                v.reasons.append("nonfinite_grad")
            if self.consecutive_bad >= c.max_consecutive_bad:
                v.action = "rollback"
                v.reasons.append(f"{self.consecutive_bad} consecutive non-finite steps")
            elif len(self.bad_steps) >= c.max_bad_in_window:
                v.action = "rollback"
                v.reasons.append(f"{len(self.bad_steps)} non-finite steps in last {c.bad_window}")
            else:
                v.action = "skip"
            return v
        self.consecutive_bad = 0

        if self.ema is not None and self.n >= c.ema_warmup:
            std = math.sqrt(max(self.ema_var, 1e-12))
            if loss > self.ema + c.spike_zscore * std and loss > self.ema * c.spike_ratio:
                v.spike = True
                self.last_incident_step = step
                self.spike_steps.append(step)
                while self.spike_steps and self.spike_steps[0] <= step - c.spike_window:
                    self.spike_steps.popleft()
                v.reasons.append(f"loss_spike loss={loss:.4f} ema={self.ema:.4f} std={std:.4f}")
                if len(self.spike_steps) >= c.spike_rollback_count:
                    v.action = "rollback"
                    v.reasons.append(f"{len(self.spike_steps)} spikes in last {c.spike_window} steps")
                elif c.skip_spike_updates:
                    v.action = "skip"
        if not v.spike:  # spikes do not contaminate the running statistics
            if self.ema is None:
                self.ema, self.ema_var = loss, 0.0
            else:
                d = loss - self.ema
                self.ema_var = c.ema_beta * (self.ema_var + (1 - c.ema_beta) * d * d)
                self.ema = self.ema + (1 - c.ema_beta) * d
            self.n += 1
        if self.gn_ema is not None and grad_norm > c.grad_spike_ratio * self.gn_ema and self.n > c.ema_warmup:
            v.reasons.append(f"grad_norm_spike gn={grad_norm:.3f} ema={self.gn_ema:.3f}")
        self.gn_ema = grad_norm if self.gn_ema is None else 0.98 * self.gn_ema + 0.02 * grad_norm
        return v
