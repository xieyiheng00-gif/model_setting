# lmarch: a controlled architecture comparison on the packed 10B-token data

Five 12-layer language models that are identical except for the token mixer in each layer. All five are
trained on the two-stage data plan in `llm_training/TRAINING_DATA.md`: a 7B-token 4K trunk, then two
stage-2 branches (16K and 4K) on the same 2.84B tokens.

| arch       | pattern                   | non-embedding* | mixer                                                           |
|------------|---------------------------|----------------|-----------------------------------------------------------------|
| `dense`    | `F x 12`                  | 85.0M          | full causal softmax attention                                   |
| `dsa`      | `S x 12`                  | 87.9M          | DeepSeek Sparse Attention (V3.2): lightning indexer + top-k     |
| `kda_full` | `K K K F K K K F K K K F` | 86.9M          | Kimi Delta Attention + full attention every 4th layer           |
| `kda_dsa`  | `K K K S K K K S K K K S` | 87.7M          | KDA + DSA every 4th layer                                       |
| `csa`      | `C x 12`                  | 89.8M          | Compressed Sparse Attention (DeepSeek-V4)                       |

\*Plus the tied embedding: 128,256 × 768 = **98.5M**, which gives about **183–188M in total**. The
data uses the Llama-3 tokenizer (128,256 ids), so the original "~110M with a 32K vocab" can't be
kept without re-tokenizing. The block stack is unchanged: d_model 768, 12 heads × 64, SwiGLU 2048.
Exact counts are printed at start-up and written to `runs/<run>/meta.json`.

## Quick start

```bash
pip install -r requirements.txt       # H100/Linux: + flash-linear-attention, flash-attn (optional)
# the data (TRAINING_DATA.md §1) and its reader: this repo imports llm_training/data_prep.loader
hf download xie1231/llm-10b-packed --repo-type dataset --local-dir /data/llm/packed
#   keep llm_training/ next to this repo, or set data.data_prep_root / $LMARCH_DATA_PREP

pytest -q tests                       # CPU; builds a synthetic packed dataset with the real schema

# SMOKE MODE: the whole plan on a 42 MB slice of the real data (RTX 3060: minutes per arch)
python scripts/make_smoke_data.py     # -> data/packed_smoke
python scripts/smoke.py               # trunk -> 16K + 4K branches for all 5 archs, then automatic checks
python scripts/smoke.py --archs dense kda_dsa --faults    # + rehearse NaN rollback and crash restart
python scripts/smoke.py --config_dir configs/rtx3060 --archs dense --stages trunk s2_4k   # full-size model

# REAL RUNS (8 x H100): stage 1, then both branches, every arch, under the crash supervisor
bash scripts/run_all.sh h100x8
# or one at a time
python scripts/supervise.py --run_dir runs/dsa_trunk_h100x8 -- \
    torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/h100x8/trunk.yaml --arch dsa
python scripts/supervise.py --run_dir runs/dsa_s2_16k_h100x8 -- \
    torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/h100x8/s2_16k.yaml --arch dsa
python scripts/analyze_runs.py runs/*_trunk_h100x8 --out reports/trunk
```

You can override any config value, e.g. `--set optim.lr=3e-4 data.packed_dir=/mnt/packed`.

## Training plan and configs

```
Stage 1  trunk_4k   4K rows, 128 rows/step (524,288 tok), steps 0 -> 13,351     LR: warmup 250 + constant
    │  branch point: runs/<arch>_trunk_<hw>/checkpoints/step_0013351_final (weights, Adam, step, RNG)
    ├── s2_16k  stage2  32 x 16K rows/step, steps 13,351 -> 18,770   LR: 1-sqrt decay to 0
    └── s2_4k   stage2  the same batches re-cut into 128 x 4K rows    LR: identical decay
```

The WSD schedule spans the whole plan: 71% of the steps are warmup plus constant LR (stage 1), and
29% decay (each branch). That is your "70% + 30%" split, with the decay happening in stage 2. Steps are
absolute: stage 2 continues the step counter. Data index 0 of a stage is its start step.

| file | stage | rows/step × seq | micro × GPUs × accum | notes |
|---|---|---|---|---|
| `configs/h100x8/trunk.yaml`  | 1        | 128 × 4096  | 4 × 8 × 4  | |
| `configs/h100x8/s2_16k.yaml` | 2, 16K   | 32 × 16384  | 1 × 8 × 4  | activation checkpointing |
| `configs/h100x8/s2_4k.yaml`  | 2, 4K    | 128 × 4096  | 4 × 8 × 4  | |
| `configs/h100x1/*.yaml`      | same     | same        | accum 32   | same data and optimization as 8 GPUs |
| `configs/rtx3060/*.yaml`     | same     | subset      | micro 1    | **full-size** model on the smoke data (memory/speed check) |
| `configs/smoke/*.yaml`       | same     | 4 × 4K / 2 × 16K / 8 × 4K | | d_model 256, 40 + 20 steps on `data/packed_smoke` |
| `configs/test/*.yaml`        | same     | re-cut short | | tiny model, synthetic data (pytest) |

Every file inherits `configs/stage/<stage>.yaml` → `configs/base.yaml`. The run name is
`<arch>_<stage>_<hw>`. A stage-2 run starts from `checkpoint.init_from` (default
`{out_dir}/{arch}_trunk_{hardware}`) when it has no checkpoint of its own. Weights and Adam moments
are matched **by name**. Parameters that are new at the branch get a fresh init and fresh optimizer
state, as TRAINING_DATA.md asks for a DSA indexer added at the branch: e.g. `--arch dsa` with
`--set checkpoint.init_from=runs/dense_trunk_h100x8` (V3.2-style conversion). For those parameters the
indexer's dense warm-up restarts at the branch. After that the run's own checkpoints take over.

## How the data is used (`lmarch/data/packed.py`)

- **Reader.** Stage 1 reads `trunk_4k` in file order, batch *s* = rows [128s, 128s+128). This is
  byte-identical to the reference `iter_trunk_batches` (verified on the real data, including resume).
  Resuming uses parquet row-group metadata, so there's no 50 s read-through. Stage 2 uses
  `data_prep.loader.iter_stage2_batches`. The 4K branch re-cuts the same batches, identical to
  `view="4k"`.
- **Ranks.** Trunk rows go to ranks as `rows[rank::world]`. Stage 2 uses `data_prep.rank_rows`, which
  gives every GPU the same long/short mix. The data per optimizer step does not depend on the number
  of GPUs.
- **Labels and loss.** Labels come from `doc_labels`, shifted for next-token prediction, so the first
  token of each segment is never a target. The loss is **averaged over the non-ignored labels of the
  global batch**. Every rank computes that count from the global batch, then scales its summed loss by
  `world / global_count`, and DDP averages the gradients.
- **Document masking** (`data.doc_mask: true`). Positions restart per segment (`position_ids`), and
  every mixer is confined to its segment:
  - F: flash-attn varlen (`cu_seqlens`) or SDPA with a document-causal mask.
  - S: top-k is taken among the segment's keys only.
  - K: the recurrent state resets at segment starts (inside chunks in the PyTorch kernel, via
    `cu_seqlens` in fla), and the short conv doesn't reach back across a boundary.
  - C: a compressed entry pools only tokens of its own document, and the window stays in the segment.
  - Tests check that a document inside a packed row gives the same outputs as the document alone,
    for every architecture.
- **128K vocabulary.** The LM head and cross-entropy run in 2,048-token chunks under checkpointing,
  so the full logits never exist.
- **Holdout eval.** Periodic evals use 16 rows per group (1.8M tokens). The end of each stage runs
  the full holdout (35M tokens). Each eval reports loss overall, per **source × row_type**
  (TRAINING_DATA.md §3), and per position inside the document. `eval.extra_seq_lens` adds
  cross-length evals: the 16K branch also evaluates at 4K, and the 4K branch at 16K. For DSA/CSA
  there is also a dense-attention comparison.
- **Smoke subset.** `scripts/make_smoke_data.py` copies the first 2 trunk row groups, 32 complete
  stage-2 steps and 8 holdout rows per group, with the same schema and manifests.

## Model design

**Block** (every layer): `x = x + Mixer(RMSNorm(x));  x = x + SwiGLU(RMSNorm(x))`

| component | choice |
|---|---|
| embedding | trainable 128,256 × 768, tied with the LM head |
| norms | RMSNorm (eps 1e-6), pre-norm, final RMSNorm; no biases |
| FFN | SwiGLU, hidden 2048 (fused gate/up) |
| positions | RoPE θ=1e4 (per-document positions) in F/S; partial RoPE in CSA and the indexers; none in KDA |
| init | N(0, 0.02); residual output projections N(0, 0.02/√24); KDA `A_log = log U(1,16)`, `dt_bias = softplus⁻¹(U_log[1e-3, 0.1])` |
| optimizer | AdamW (0.9, 0.95), eps 1e-8, wd 0.1 on matrices only, clip 1.0, peak LR 6e-4 (sweep it) |
| DSA | 4 × 64 indexer, top-k **512** (same at 4K and 16K); V3.2 recipe: detached indexer input, KL to head-averaged attention, 500-step dense warm-up |
| KDA | channel-wise gated delta rule, short conv 4, L2-normalized q/k, low-rank gates, fla kernel when it passes the self-check |
| CSA | m=4 overlapping gated compression, 6 shared-KV heads, indexer top-**128** blocks + 128-token window + sink, partial RoPE with inverse RoPE on the output |

> **Throughput caveat.** DSA/CSA are computed exactly, but with boolean masks on dense kernels. At
> 16K they build (T × T) masks. Compare their quality metrics; don't read their tokens/sec and memory
> as architectural numbers.

## What is saved

Metrics go to `runs/<run>/metrics/*.jsonl` (append-only; readers keep the last record per step) and, live,
to W&B (next section),
checkpoints to `checkpoints/`, NaN forensics to `crash_reports/`, plus `meta.json`, `config.yaml` and
`heartbeat.json`.

- **Every step:** `loss` (global-token-weighted), `lr`, `grad_norm`, **`loss_nonfinite` /
  `grad_nonfinite`**, `step_time`, `tokens_per_sec`, `peak_mem_gb`, `skipped/action/reasons`, plus:
  - `loss_by_source`, `loss_by_row_type`, `valid_tokens`, `long_token_frac`, `mean_segment_len`;
  - `data_index`, `data_batch`, `data_epoch`, indexer KL per layer and type.
- **Every 100 steps**, per layer and **grouped by layer type**:
  - activation RMS at each block output, per-group grad norms, and update/weight ratios with a health
    label and alerts;
  - F/S/C: max logit and entropy per head;
  - S/C: indexer KL, attention-mass recall, effective sparsity and density;
  - C: sink/window mass;
  - K: forget-gate, β and state-norm distributions;
  - LM-logit max/std/logsumexp (chunked).
  - The probe batch is a fixed mix of holdout rows.
- **Evals:** holdout loss overall, by group, by in-document position, cross-length, and dense vs
  sparse.

## Live monitoring with Weights & Biases

W&B is on by default (`log.wandb: true`, project `model_setting`). Rank 0 mirrors the JSONL logs, which
stay the source of truth.

**API key: never put it in a config, a script or the repo.** Authenticate each machine once:
```bash
pip install wandb
wandb login          # paste the key from https://wandb.ai/authorize; stored in ~/.netrc (_netrc on Windows)
```
On a cluster, `export WANDB_API_KEY=...` from a secret works instead. With no key, the run logs
offline to `runs/<run>/wandb/` and prints the `wandb sync` command to upload it later. W&B problems
(no package, auth, network) switch W&B off with a message; they never stop training.

- **Grouping:**
  - run name = `<arch>_<stage>_<hw>`;
  - group = `{stage}_{hardware}` (e.g. `trunk_h100x8`), so the five architectures of a stage sit together;
  - `job_type` = stage;
  - tags: arch, stage, hardware and layer pattern, plus `log.wandb_tags`.
- **Panels:**
  - `train/*`, `train_source/*`, `train_rowtype/*` every step, including the NaN/Inf flags;
  - `diag_<type>/*` every 100 steps, per layer type (mean and `.max` over the layers of that type);
  - `layer_<metric>/Lxx_<type>` per layer (activation RMS, update ratios, grad norm, max logit,
    recall, forget gate, β, KDA state norm, sink mass);
  - `eval/*`, `eval_group/*`, `eval_pos/*`, and `eval_full*` for the end-of-stage full holdout;
  - `events/*` counters (rollbacks, NaN forensics, ratio alerts, resumes, checkpoints).
  - x-axis: `trainer/step`.
- **Crashes and rollbacks:** the W&B run ID is stored in `runs/<run>/wandb_run_id.txt`, so supervisor
  restarts continue the same W&B run. Steps re-run after a rollback are logged again, and
  `train/segment` increases.
- **Alerts** (email or Slack, per your W&B settings): rollback, divergence, OOM, crash, persistent
  update-ratio alerts (at most every 30 min), and stage finished.
- **Switches:**
  - `--set log.wandb=false` turns it off;
  - `log.wandb_mode=offline` suits nodes without internet;
  - `log.wandb_entity=<team>` logs to a team;
  - `log.wandb_log_layers=false` reduces the panels.

## Crash handling

The trainer skips non-finite steps and writes a forensics report naming the first non-finite module
and its layer type. Repeated failures trigger a **rollback** to the last healthy checkpoint of the
stage. The rollback skips the offending data by raising `data_skip`, so step *s* then reads batch
*s − start + data_skip*. It escalates to older checkpoints, and after `max_rollbacks` it exits with
code 4. `scripts/supervise.py` restarts on errors and preemption, halves the micro-batch on OOM, and
kills hangs (no heartbeat for 30 min). Checkpoints are atomic (`COMPLETE` marker) and include
optimizer parameter names, which makes the branch point robust to model changes. If rollbacks make a
stage run past the end of its data, the stream wraps around (epoch + 1) and logs a warning. The table
mapping each symptom to its diagnostic and fix is unchanged: max logit growing → `qk_norm`/lower LR;
recall low → longer dense warm-up; KDA state norm exploding → lower LR; `logit_lse_mean` drifting →
`z_loss_coef`.

`python scripts/smoke.py --faults` rehearses all of this on real data:
poisoned batches → skip → rollback → recovery, then an injected crash → supervisor restart → exact resume.

## Layout

```
lmarch/config.py        hyperparameters (dataclasses), YAML inherit (lists allowed), --set overrides
lmarch/data/packed.py   packed-data reader: trunk/stage-2 streams, labels, rank split, holdout, prefetch
lmarch/data/synthetic.py  synthetic data with the real schema (tests)
lmarch/model/           common (DocInfo, RoPE, norms), attention (F, S), indexer, kda, csa, model (chunked CE)
lmarch/train/           trainer, guard, monitor, checkpoint (+ branch loading), logger, optim (WSD), dist
scripts/                train, supervise, smoke, make_smoke_data, analyze_runs, run_all.{sh,ps1}
configs/                base, stage/{trunk,s2_16k,s2_4k}, {h100x8,h100x1,rtx3060,smoke,test}/*
```
