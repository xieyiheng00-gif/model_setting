"""Diagnostics collected every `diag.interval` steps and their aggregation by layer type.

Per layer (and per parameter group: mixer / indexer / ffn / norm):
  grad norm (before clipping), weight norm, update/weight ratio ||dW||/||W|| of the weight matrices
  (+ health label; 1-D parameters separately as vec_ratio), activation RMS at the
  block output (+ mixer/FFN branch RMS, abs-max), and the mixer-specific statistics:
    softmax/DSA/CSA: max pre-softmax logit and entropy per head
    DSA/CSA:         indexer KL, attention-mass recall, effective sparsity, density (+ sink / window mass for CSA)
    KDA:             forget-gate distribution, beta distribution, recurrent state norm at sequence end
Everything is also reduced per layer type (softmax / dsa / kda / csa) so hybrids show which mechanism
misbehaves.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch

from ..config import DiagConfig, LAYER_TYPE_NAMES

GROUPS = ("mixer", "indexer", "ffn", "norm")


class ParamIndex:
    """Static index of parameters by (layer_key, group) for fast grouped norms."""

    def __init__(self, raw_model):
        self.entries = raw_model.param_groups_by_layer()           # (name, p, layer_key, group)
        self.layer_types = {f"L{i:02d}": LAYER_TYPE_NAMES[c] for i, c in enumerate(raw_model.pattern)}
        self.keys = sorted({(k, g) for _, _, k, g in self.entries})

    @torch.no_grad()
    def grad_norms(self) -> dict:
        sq = defaultdict(float)
        ps = [(k, g, p.grad) for _, p, k, g in self.entries if p.grad is not None]
        if not ps:
            return {}
        norms = torch._foreach_norm([t for _, _, t in ps])
        vals = torch.stack([n.float() for n in norms]).cpu().tolist()
        for (k, g, _), v in zip(ps, vals):
            sq[(k, g)] += v * v
        return {kg: math.sqrt(s) for kg, s in sq.items()}

    @torch.no_grad()
    def snapshot(self) -> list[torch.Tensor]:
        return [p.detach().clone() for _, p, _, _ in self.entries]

    @torch.no_grad()
    def update_stats(self, snap: list[torch.Tensor]) -> dict:
        """-> {(layer, group): {"ratio", "w_norm", "vec_ratio", "max_matrix_ratio", "max_matrix"}}.

        ratio: ||dW||/||W|| over the weight MATRICES of the group (the quantity the ~1e-3 rule of thumb is
        about). 1-D parameters (norm gains, biases, KDA A_log/dt_bias, CSA positional biases/sink) have large
        norms of their own (e.g. dt_bias ~ -4.6 per channel) and would swamp it; they are reported separately
        as vec_ratio. A group without matrices (norm) falls back to its vectors.
        w_norm: norm of ALL parameters of the group."""
        cur = [p.detach() for _, p, _, _ in self.entries]
        deltas = torch._foreach_sub(cur, snap)
        d_n = torch.stack([n.float() for n in torch._foreach_norm(deltas)]).cpu().tolist()
        w_n = torch.stack([n.float() for n in torch._foreach_norm(snap)]).cpu().tolist()
        agg: dict = defaultdict(lambda: {"md2": 0.0, "mw2": 0.0, "vd2": 0.0, "vw2": 0.0,
                                         "max_matrix_ratio": 0.0, "max_matrix": ""})
        for (name, p, k, g), dn, wn in zip(self.entries, d_n, w_n):
            a = agg[(k, g)]
            if p.ndim >= 2:
                a["md2"] += dn * dn
                a["mw2"] += wn * wn
                r = dn / wn if wn > 0 else 0.0
                if r > a["max_matrix_ratio"]:
                    a["max_matrix_ratio"], a["max_matrix"] = r, name
            else:
                a["vd2"] += dn * dn
                a["vw2"] += wn * wn
        ratio = lambda d2, w2: math.sqrt(d2) / max(math.sqrt(w2), 1e-12)
        out = {}
        for kg, a in agg.items():
            has_m = a["mw2"] > 0
            out[kg] = {"ratio": ratio(a["md2"], a["mw2"]) if has_m else ratio(a["vd2"], a["vw2"]),
                       "w_norm": math.sqrt(a["mw2"] + a["vw2"]),
                       "max_matrix_ratio": a["max_matrix_ratio"], "max_matrix": a["max_matrix"]}
            if has_m and a["vw2"] > 0:
                out[kg]["vec_ratio"] = ratio(a["vd2"], a["vw2"])
        return out


def health(ratio: float, cfg: DiagConfig) -> str:
    if ratio > cfg.ratio_high:
        return "high"
    if ratio < cfg.ratio_low:
        return "low"
    return "ok"


class RatioAlerts:
    """Fires once a (layer, group) stays high/low for `ratio_persist` consecutive diagnostic checks.
    Nothing fires before `alert_start_step`: right after warm-up every layer (the embedding most of all)
    still sits near its N(0, 0.02) init, so Adam steps of ~lr per entry give ratios of ~1e-2 that shrink
    by themselves as the weights grow. The ratios are logged all the same."""

    def __init__(self, cfg: DiagConfig):
        self.cfg = cfg
        self.streak: dict = {}

    def update(self, stats: dict, lr_frac: float, step: int) -> list[dict]:
        alerts = []
        for (layer, grp), s in stats.items():
            if grp not in ("mixer", "indexer", "ffn", "embed"):
                continue
            h = health(s["ratio"], self.cfg)
            if step < self.cfg.alert_start_step:
                h = "ok"   # early training: high ratios are expected (see class docstring)
            if h == "low" and lr_frac < self.cfg.alert_min_lr_frac:
                h = "ok"   # LR is being annealed to ~0: small updates are expected
            prev_h, n = self.streak.get((layer, grp), ("ok", 0))
            n = n + 1 if (h == prev_h and h != "ok") else (1 if h != "ok" else 0)
            self.streak[(layer, grp)] = (h, n)
            if h != "ok" and n == self.cfg.ratio_persist:
                alerts.append({"layer": layer, "group": grp, "state": h, "ratio": s["ratio"],
                               "message": f"{layer}/{grp} update ratio {s['ratio']:.2e} {h} for "
                                          f"{n} consecutive checks ("
                                          + ("LR too high for this layer" if h == "high" else "barely learning")
                                          + ")"})
        return alerts

    def state_dict(self) -> dict:
        return {f"{k[0]}|{k[1]}": list(v) for k, v in self.streak.items()}

    def load_state_dict(self, d: dict) -> None:
        self.streak = {tuple(k.split("|")): tuple(v) for k, v in d.items()}


# --------------------------------------------------------------------------------------
# Aggregation by layer type
# --------------------------------------------------------------------------------------
def _numeric_leaves(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v is not None and math.isfinite(float(v)):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(_numeric_leaves(v, key + "."))
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            # per-head lists: reduce to mean/min/max so they can be compared across layer types
            out[key + ".mean"] = sum(v) / len(v)
            out[key + ".min"] = min(v)
            out[key + ".max"] = max(v)
    return out


def group_by_type(layers: list[dict]) -> dict:
    """layers: [{"type": ..., <nested numeric stats>}] -> {type: {metric: {mean,min,max,n}}}."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for L in layers:
        for k, v in _numeric_leaves({kk: vv for kk, vv in L.items() if kk not in ("idx", "type")}).items():
            acc[L["type"]][k].append(v)
    out = {}
    for t, metrics in acc.items():
        out[t] = {k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)}
                  for k, v in metrics.items()}
    return out


@torch.no_grad()
def collect_probe(raw_model) -> list[dict]:
    """After a diag forward: gather block + mixer statistics and clear them."""
    layers = []
    for b in raw_model.blocks:
        d = {"idx": b.idx, "type": b.type_name}
        if b.last_diag:
            d["act"] = b.last_diag
        if getattr(b.mixer, "last_diag", None):
            d["mixer"] = b.mixer.last_diag
        b.last_diag = None
        b.mixer.last_diag = None
        layers.append(d)
    return layers
