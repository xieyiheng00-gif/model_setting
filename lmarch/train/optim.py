"""AdamW parameter groups and the Warmup-Stable-Decay learning-rate schedule (absolute steps)."""
from __future__ import annotations

import math

import torch

from ..config import OptimConfig, ScheduleConfig


def _no_decay(name: str, p: torch.Tensor) -> bool:
    """Weight decay only on matrices of Linear layers. Excluded: embeddings (tied LM head), norms,
    biases, KDA A_log/dt_bias, CSA positional biases and sink, depthwise conv kernels."""
    if p.ndim < 2:
        return True
    return any(s in name for s in ("embed", "lm_head", "norm", "pos_bias", "conv", "A_log", "dt_bias", "sink"))


def build_optimizer(model: torch.nn.Module, ocfg: OptimConfig, indexer_lr_mult: float,
                    device: torch.device) -> torch.optim.AdamW:
    """AdamW with groups (main|indexer) x (decay|no_decay). `opt.param_names` lists the parameter names
    in optimizer order (used to map optimizer state by name at a branch point)."""
    groups: dict = {}
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        is_idx = ".indexer." in name or ".idx_" in name
        nd = _no_decay(name, p)
        key = ("indexer" if is_idx else "main", "no_decay" if nd else "decay")
        g = groups.setdefault(key, {"params": [], "names": [],
                                    "weight_decay": 0.0 if nd else ocfg.weight_decay,
                                    "lr_mult": indexer_lr_mult if is_idx else 1.0,
                                    "group_name": f"{key[0]}/{key[1]}"})
        g["params"].append(p)
        g["names"].append(name)
    param_groups, names = [], []
    for g in groups.values():
        g = dict(g)
        names += g.pop("names")
        param_groups.append(g)
    fused = ocfg.fused and device.type == "cuda"
    opt = torch.optim.AdamW(param_groups, lr=ocfg.lr, betas=(ocfg.beta1, ocfg.beta2), eps=ocfg.eps, fused=fused)
    opt.param_names = names
    return opt


def lr_multiplier(step: int, max_steps: int, s: ScheduleConfig) -> float:
    """WSD over absolute steps: linear warmup -> constant -> decay from `decay_start` to `max_steps`.
    Stage 1 (trunk) has no decay; each stage-2 branch decays from the branch point."""
    if step < s.warmup_steps:
        return (step + 1) / max(1, s.warmup_steps)
    start = s.decay_start if s.decay_start >= 0 else int(round(max_steps * (1 - s.decay_frac)))
    if step < start or start >= max_steps:
        return 1.0
    p = min(1.0, (step - start) / (max_steps - start))
    if s.decay_shape == "linear":
        f = 1.0 - p
    elif s.decay_shape == "cosine":
        f = 0.5 * (1.0 + math.cos(math.pi * p))
    else:  # "1-sqrt" (Hägele et al., 2024)
        f = 1.0 - math.sqrt(p)
    return s.min_lr_ratio + (1.0 - s.min_lr_ratio) * f
