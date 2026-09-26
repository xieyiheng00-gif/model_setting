"""Configuration: dataclasses, YAML loading (with `inherit:`), CLI overrides, validation.

Every knob of a run lives here so that a run directory's `config.yaml` fully describes it.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Layer codes: F = full softmax attention, S = DeepSeek Sparse Attention (DSA),
#              K = Kimi Delta Attention (KDA), C = Compressed Sparse Attention (CSA, DeepSeek-V4).
ARCH_PRESETS: dict[str, str] = {
    "dense": "F" * 12,
    "dsa": "S" * 12,
    "kda_full": "KKKF" * 3,
    "kda_dsa": "KKKS" * 3,
    "csa": "C" * 12,
}
LAYER_TYPE_NAMES = {"F": "softmax", "S": "dsa", "K": "kda", "C": "csa"}
INDEXER_TYPES = ("S", "C")


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
@dataclass
class IndexerTrainConfig:
    """How the lightning indexers of DSA/CSA layers are trained (DeepSeek-V3.2 recipe)."""
    dense_warmup_steps: int = 500   # steps of dense attention while the indexer learns via KL
    detach_input: bool = True       # indexer sees h.detach(): trained only by the KL loss
    loss_coef: float = 1.0          # weight of the KL loss (its gradient only reaches the indexer)
    kl_max_rows: int = 0            # compute KL on at most this many sequences per micro-batch (0 = all)
    lr_mult: float = 1.0            # LR multiplier for indexer parameters


@dataclass
class DSAConfig:
    top_k: int = 512                # tokens attended per query in the sparse phase (same in 4K and 16K)
    index_heads: int = 4
    index_dim: int = 64
    index_rope_dim: int = 32        # partial RoPE inside the indexer


@dataclass
class CSAConfig:
    compress_ratio: int = 4         # m: every m tokens -> one compressed KV entry (windows of 2m overlap)
    kv_heads: int = 6               # shared-KV heads (K = V entry, as in V4); 6 matches F's K/V param count
    top_k: int = 128                # compressed entries per query (128 x 4 = 512 tokens of coverage)
    window: int = 128               # sliding window of uncompressed tokens
    rope_dim: int = 32              # partial RoPE on the last `rope_dim` channels (+ inverse RoPE on output)
    attn_sink: bool = True          # learnable per-head sink logit in the softmax denominator
    index_heads: int = 4
    index_dim: int = 64
    index_rope_dim: int = 32


@dataclass
class KDAConfig:
    conv_size: int = 4              # causal depthwise short conv on q/k/v
    gate_rank: int = 64             # low-rank bottleneck of forget-gate and output-gate projections
    backend: str = "auto"           # auto | fla | torch   (auto = fla if importable AND passes self-check)
    chunk_size: int = 32            # chunk length of the pure-PyTorch fallback kernel
    A_init_min: float = 1.0
    A_init_max: float = 16.0
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4


@dataclass
class ModelConfig:
    arch: str = "dense"
    layer_pattern: str = ""         # explicit pattern (e.g. "KKKFKKKFKKKF"); overrides the arch preset
    vocab_size: int = 128256        # Llama-3 tokenizer of the packed data (ids 0..128001)
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    ffn_dim: int = 2048
    max_seq_len: int = 4096         # raised automatically to the longest train/eval length
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    qk_norm: bool = False           # optional QK-RMSNorm for F/S layers (off: keep logit growth observable)
    tie_embeddings: bool = True     # tied: ~85M non-embedding + 98.5M embedding (128K vocab)
    init_std: float = 0.02
    hybrid_softmax_nope: bool = False  # Kimi-Linear style: no RoPE in F/S layers of KDA hybrids
    attn_backend: str = "auto"      # auto | sdpa | flash_varlen  (document-masked full attention)
    dsa: DSAConfig = field(default_factory=DSAConfig)
    csa: CSAConfig = field(default_factory=CSAConfig)
    kda: KDAConfig = field(default_factory=KDAConfig)
    indexer: IndexerTrainConfig = field(default_factory=IndexerTrainConfig)

    def pattern(self) -> str:
        if self.layer_pattern:
            return self.layer_pattern.replace(" ", "").upper()
        if self.arch not in ARCH_PRESETS:
            raise ConfigError(f"unknown arch '{self.arch}', choose from {sorted(ARCH_PRESETS)}")
        return ARCH_PRESETS[self.arch]


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    seq_len: int = 4096             # rows longer than this are re-cut (4K view of stage 2, smoke tests)
    micro_batch_size: int = 4       # rows per GPU per micro-step; grad-accum = rows_per_step/(micro*world)
    max_steps: int = 13351          # ABSOLUTE final step (stage 2 continues the stage-1 step counter)
    seed: int = 1337
    dtype: str = "bf16"             # bf16 (autocast, fp32 master weights) | fp32
    compile: bool = False
    activation_checkpointing: bool = False
    grad_clip: float = 1.0
    z_loss_coef: float = 0.0
    loss_chunk_tokens: int = 2048   # LM head + CE computed in chunks (128K vocab: never materialise all logits)
    tf32: bool = True
    nccl_timeout_min: int = 30


@dataclass
class OptimConfig:
    lr: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    fused: bool = True


@dataclass
class ScheduleConfig:
    """Warmup-Stable-Decay over ABSOLUTE steps. Stage 1 (trunk) = warmup + constant (decay_frac 0);
    each stage-2 branch decays from `decay_start` (the branch point) to `train.max_steps`."""
    warmup_steps: int = 250
    decay_frac: float = 0.0
    decay_start: int = -1           # absolute step where decay begins; -1 = max_steps * (1 - decay_frac)
    decay_shape: str = "1-sqrt"     # linear | cosine | 1-sqrt
    min_lr_ratio: float = 0.0


@dataclass
class DataConfig:
    """Packed Llama-3 data from llm_training/data_prep (see TRAINING_DATA.md)."""
    packed_dir: str = "D:/data/llm/packed"   # contains trunk_4k/, stage2/, holdout/
    data_prep_root: str = ""        # folder containing data_prep/ ("" = importable / $LMARCH_DATA_PREP / ../llm_training)
    stage: str = "trunk"            # trunk | stage2
    stage2_view: str = "16k"        # 16k | 4k   (4k = the same batches re-cut into 4K rows)
    trunk_rows_per_step: int = 128  # 128 x 4K = 524,288 tokens (stage 2 is fixed by the data: 32 x 16K)
    stage_start_step: int = -1      # absolute step of data index 0; -1 = 0 from scratch, branch step for init_from
    doc_mask: bool = True           # attention / recurrence / conv never cross document segments
    max_rows_per_step: int = 0      # smoke tests: keep only this many rows of each global batch (mix-preserving)
    check_token_ids: bool = True    # verify ids < model.vocab_size on every batch
    prefetch: int = 2               # optimizer steps prepared ahead by the background thread


@dataclass
class EvalConfig:
    """Holdout: 7 groups (source x row_type) x 305 rows of 16K tokens."""
    interval: int = 500
    rows_per_group: int = 16        # periodic eval: first N rows of every group (0 = all 305)
    final_full: bool = True         # full holdout (35M tokens) at the end of the stage
    extra_seq_lens: list = field(default_factory=list)   # also evaluate at these lengths (e.g. [16384])
    micro_batch_size: int = 0       # 0 = train.micro_batch_size
    position_buckets: list = field(default_factory=lambda: [64, 256, 1024, 4096, 8192])
    dense_compare: bool = True      # for DSA/CSA also evaluate with dense attention (sparsity cost)


@dataclass
class DiagConfig:
    interval: int = 100
    probe_batch_size: int = 4       # probe rows from the holdout (mixed sources), at train.seq_len
    ratio_high: float = 1e-2        # ||dW||/||W|| persistently above -> LR too high for that layer
    ratio_low: float = 1e-4         # below -> layer is barely learning
    ratio_persist: int = 3          # consecutive diagnostic checks before an alert fires
    alert_start_step: int = 1000    # no ratio alerts before this step (near-init weights give ~1e-2 ratios)
    alert_min_lr_frac: float = 0.1  # suppress "barely learning" alerts once LR decayed below this fraction


@dataclass
class CheckpointConfig:
    interval: int = 250
    keep_last: int = 3
    milestone_interval: int = 1000  # these are never deleted
    resume: str = "auto"            # auto | none | <path to a checkpoint dir>
    save_final: bool = True
    init_from: str = ""             # branch start when the run has no checkpoint yet: a checkpoint dir or a
                                    # run dir (its *_final / newest). {arch} {hardware} {out_dir} are substituted.
    init_allow_new_params: bool = True  # e.g. a DSA indexer added at the branch: fresh init + fresh Adam state


@dataclass
class GuardConfig:
    """Divergence handling: skip bad steps, detect loss spikes, roll back to the last healthy checkpoint."""
    max_consecutive_bad: int = 3    # consecutive non-finite steps before rollback
    bad_window: int = 100
    max_bad_in_window: int = 8      # non-finite steps inside `bad_window` before rollback
    ema_beta: float = 0.98
    ema_warmup: int = 100
    spike_zscore: float = 6.0
    spike_ratio: float = 1.15       # also require loss > ratio * EMA
    skip_spike_updates: bool = False
    spike_window: int = 50
    spike_rollback_count: int = 5   # spikes inside `spike_window` before rollback
    grad_spike_ratio: float = 10.0  # grad-norm spike (logged only)
    rollback_skip_extra: int = 20   # batches skipped beyond the failing step after a rollback
    lr_scale_on_rollback: float = 1.0  # multiply LR by this after every rollback (1.0 = unchanged)
    max_rollbacks: int = 5
    max_rollbacks_per_checkpoint: int = 2  # then go back to the next older healthy checkpoint
    healthy_window: int = 20        # a checkpoint is "healthy" if no bad step / spike in this many steps
    forensics: bool = True
    max_forensics: int = 5


@dataclass
class LogConfig:
    out_dir: str = "runs"
    run_name: str = ""
    console_interval: int = 10
    tensorboard: bool = False
    wandb: bool = False             # mirror metrics to Weights & Biases (rank 0); key via `wandb login`, never here
    wandb_project: str = "lmarch"
    wandb_entity: str = ""          # W&B user or team; "" = your default entity
    wandb_group: str = "{stage}_{hardware}"   # runs compared together; {arch} {stage} {hardware} substituted
    wandb_tags: list = field(default_factory=list)   # extra tags (arch, stage, hardware are always added)
    wandb_mode: str = "online"      # online | offline | disabled  (online without a key -> offline)
    wandb_alerts: bool = True       # W&B alerts on rollback / divergence / OOM / crash / ratio alerts / stage end
    wandb_log_layers: bool = True   # per-layer panels (layer_<metric>/Lxx_<type>) in addition to per-type ones
    heartbeat_interval_sec: float = 30.0


@dataclass
class DebugConfig:
    """Fault injection to rehearse the crash-handling path (keep off for real runs).
    NaN/spike injection is keyed on the DATA index of the step (its batch in the stage's data stream),
    i.e. it poisons specific batches; a rollback that skips them recovers."""
    inject_nan_steps: list = field(default_factory=list)   # data steps whose loss becomes NaN
    inject_spike_steps: list = field(default_factory=list)  # data steps whose loss is multiplied by 10
    inject_exception_step: int = -1


@dataclass
class Config:
    hardware: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    diag: DiagConfig = field(default_factory=DiagConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    log: LogConfig = field(default_factory=LogConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _deep_merge(dst: dict, src: dict) -> dict:
    out = copy.deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    parents = d.pop("inherit", [])
    if isinstance(parents, str):
        parents = [parents]
    base: dict = {}
    for p in parents:
        base = _deep_merge(base, load_yaml((path.parent / p).resolve()))
    return _deep_merge(base, d)


def _coerce(cur: Any, v: Any, key: str) -> Any:
    if v is None:
        return None
    try:
        if isinstance(cur, bool):
            if isinstance(v, str):
                if v.lower() in ("1", "true", "yes", "on"):
                    return True
                if v.lower() in ("0", "false", "no", "off"):
                    return False
                raise ValueError(v)
            return bool(v)
        if isinstance(cur, int):
            if isinstance(v, str):
                v = float(v)
            if isinstance(v, float) and not v.is_integer():
                raise ValueError(f"expected an integer, got {v}")
            return int(v)
        if isinstance(cur, float):
            return float(v)
        if isinstance(cur, (list, tuple)):
            if not isinstance(v, (list, tuple)):
                raise ValueError(f"expected a list, got {v!r}")
            return list(v)
        if isinstance(cur, str):
            return str(v)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"bad value for '{key}': {v!r} ({e})") from None
    return v


def apply_dict(dc: Any, d: dict, prefix: str = "") -> None:
    names = {f.name for f in fields(dc)}
    for k, v in d.items():
        if k not in names:
            raise ConfigError(f"unknown config key '{prefix}{k}'")
        cur = getattr(dc, k)
        if is_dataclass(cur):
            if not isinstance(v, dict):
                raise ConfigError(f"'{prefix}{k}' must be a mapping")
            apply_dict(cur, v, prefix + k + ".")
        else:
            setattr(dc, k, _coerce(cur, v, prefix + k))


def parse_override(s: str) -> dict:
    """'optim.lr=3e-4' -> {'optim': {'lr': 3e-4}} (value parsed as YAML)."""
    if "=" not in s:
        raise ConfigError(f"override must look like key.sub=value, got '{s}'")
    key, val = s.split("=", 1)
    node: dict = {}
    cur = node
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = yaml.safe_load(val)
    return node


def build_config(config_path: str | None = None, arch: str | None = None,
                 run_name: str | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    raw: dict = load_yaml(config_path) if config_path else {}
    for o in overrides or []:
        raw = _deep_merge(raw, parse_override(o))
    apply_dict(cfg, raw)
    if arch:
        cfg.model.arch = arch
        cfg.model.layer_pattern = ""
    if run_name:
        cfg.log.run_name = run_name
    if not cfg.log.run_name:
        cfg.log.run_name = f"{cfg.model.arch}_{stage_tag(cfg)}_{cfg.hardware or 'run'}"
    cfg.checkpoint.init_from = cfg.checkpoint.init_from.format(arch=cfg.model.arch, hardware=cfg.hardware,
                                                               out_dir=cfg.log.out_dir)
    cfg.log.wandb_group = cfg.log.wandb_group.format(arch=cfg.model.arch, hardware=cfg.hardware or "run",
                                                     stage=stage_tag(cfg))
    validate(cfg)
    return cfg


def stage_tag(cfg: Config) -> str:
    return "trunk" if cfg.data.stage == "trunk" else f"s2_{cfg.data.stage2_view}"


def validate(cfg: Config) -> None:
    m = cfg.model
    pat = m.pattern()
    if len(pat) != m.n_layers:
        raise ConfigError(f"layer pattern '{pat}' has {len(pat)} layers but n_layers={m.n_layers}")
    bad = set(pat) - set(LAYER_TYPE_NAMES)
    if bad:
        raise ConfigError(f"unknown layer codes {bad} in pattern '{pat}' (allowed: F S K C)")
    if m.d_model != m.n_heads * m.head_dim:
        raise ConfigError(f"d_model ({m.d_model}) must equal n_heads*head_dim ({m.n_heads}*{m.head_dim})")
    if m.head_dim % 2:
        raise ConfigError("head_dim must be even (RoPE)")
    for name, r in (("csa.rope_dim", m.csa.rope_dim), ("dsa.index_rope_dim", m.dsa.index_rope_dim),
                    ("csa.index_rope_dim", m.csa.index_rope_dim)):
        if r % 2:
            raise ConfigError(f"{name} must be even")
    if m.csa.rope_dim > m.head_dim or m.dsa.index_rope_dim > m.dsa.index_dim \
            or m.csa.index_rope_dim > m.csa.index_dim:
        raise ConfigError("rope dims must not exceed the corresponding head dims")
    if m.n_heads % m.csa.kv_heads:
        raise ConfigError("n_heads must be divisible by csa.kv_heads")
    if m.kda.backend not in ("auto", "fla", "torch"):
        raise ConfigError("model.kda.backend must be auto|fla|torch")
    if cfg.log.wandb_mode not in ("online", "offline", "disabled"):
        raise ConfigError("log.wandb_mode must be online|offline|disabled")
    if m.attn_backend not in ("auto", "sdpa", "flash_varlen"):
        raise ConfigError("model.attn_backend must be auto|sdpa|flash_varlen")
    t, dc, ev = cfg.train, cfg.data, cfg.eval
    if dc.stage not in ("trunk", "stage2"):
        raise ConfigError("data.stage must be trunk|stage2")
    if dc.stage2_view not in ("16k", "4k"):
        raise ConfigError("data.stage2_view must be 16k|4k")
    row_len = data_row_len(cfg)
    if row_len % t.seq_len:
        raise ConfigError(f"train.seq_len {t.seq_len} must divide the data row length {row_len}")
    for L in ev.extra_seq_lens:
        if 16384 % int(L):
            raise ConfigError(f"eval.extra_seq_lens: {L} must divide the 16384-token holdout rows")
    m.max_seq_len = max([m.max_seq_len, t.seq_len] + [int(L) for L in ev.extra_seq_lens])
    if t.dtype not in ("bf16", "fp32"):
        raise ConfigError("train.dtype must be bf16 or fp32")
    s = cfg.schedule
    if s.decay_shape not in ("linear", "cosine", "1-sqrt"):
        raise ConfigError("schedule.decay_shape must be linear|cosine|1-sqrt")
    if not 0.0 <= s.decay_frac <= 1.0:
        raise ConfigError("schedule.decay_frac must be in [0, 1]")
    if decay_start(cfg) > t.max_steps:
        raise ConfigError("schedule.decay_start is after train.max_steps")
    if 0 < decay_start(cfg) < s.warmup_steps:
        raise ConfigError("warmup must end before the decay phase starts")


def data_row_len(cfg: Config) -> int:
    """Row length delivered by the data before re-cutting to train.seq_len."""
    if cfg.data.stage == "trunk":
        return 4096
    return 16384 if cfg.data.stage2_view == "16k" else 4096


def decay_start(cfg: Config) -> int:
    s = cfg.schedule
    return s.decay_start if s.decay_start >= 0 else int(round(cfg.train.max_steps * (1 - s.decay_frac)))


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def config_from_dict(d: dict) -> Config:
    cfg = Config()
    apply_dict(cfg, d)
    return cfg
