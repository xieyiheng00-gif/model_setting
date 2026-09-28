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
pip install -r requirements.txt       # H100/Linux: + flash-linear-attention and flash-attn (see "GPU kernels")
# the data (TRAINING_DATA.md §1) and its reader: this repo imports llm_training/data_prep.loader
hf download xie1231/llm-10b-packed --repo-type dataset --local-dir /data/llm/packed
#   keep llm_training/ next to this repo, or set data.data_prep_root / $LMARCH_DATA_PREP

pytest -q tests                       # CPU; builds a synthetic packed dataset with the real schema

# SMOKE MODE: the whole plan on a 42 MB slice of the real data (RTX 3060: minutes per arch)
python scripts/make_smoke_data.py     # -> data/packed_smoke
python scripts/smoke.py               # trunk -> 16K + 4K branches for all 5 archs, then automatic checks
python scripts/smoke.py --archs dense kda_dsa --faults    # + rehearse NaN rollback and crash restart
python scripts/smoke.py --config_dir configs/rtx3060 --archs dense --stages trunk s2_4k   # full-size model

# REAL RUNS (8 x H100): follow "Checklist: training on a rented 8×H100" below. The run itself:
bash scripts/start_training.sh        # kda_full: trunk -> both branches, in a detached tmux session
bash scripts/run_all.sh h100x8        # every arch, in the foreground
# or one at a time
python scripts/supervise.py --run_dir runs/dsa_trunk_h100x8 -- \
    torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/h100x8/trunk.yaml --arch dsa
python scripts/supervise.py --run_dir runs/dsa_s2_16k_h100x8 -- \
    torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/h100x8/s2_16k.yaml --arch dsa
python scripts/analyze_runs.py runs/*_trunk_h100x8 --out reports/trunk
```

You can override any config value, e.g. `--set optim.lr=3e-4 data.packed_dir=/mnt/packed`.

## Checklist: training on a rented 8×H100

Everything to do, in order; tick the boxes as you go. The example trains **kda_full**: the trunk, then the
16K and 4K branches. That is about 7 h on 8×H100 SXM without the speed-ups (step C2 measures the real
speed). `gpu` below stands for your SSH host.

### A. On your computer, before renting
- [ ] A1. `main` has everything: `git pull`.
- [ ] A2. The keys work: `python scripts/setup_secrets.py --verify` prints your W&B, Hugging Face and GitHub
      accounts. What each key needs:
  - Hugging Face: read access to `xie1231/llm-10b-packed`, write access to
    `xie1231/model_setting-checkpoints`;
  - GitHub: read access to `model_setting` and `llm_training`.
- [ ] A3. Rent 8×H100 **SXM**, with NVLink between all GPUs and at least 100 GB of free disk (data
      16.5 GB, checkpoints about 47 GB).

### B. Set up the machine (about 15 min, mostly the data download)
- [ ] B1. Send the keys into the machine's memory, not its disk. From Git Bash in this repo:
  ```bash
  ssh gpu "umask 077 && tr -d '\r' > /dev/shm/lmarch.env" < secrets.env
  ```
- [ ] B2. Log in and do the setup inside tmux, so a dropped connection doesn't interrupt an install:
      `ssh gpu`, then `tmux new -s setup` (`apt-get install -y tmux` if it's missing).
- [ ] B3. Clone both repos side by side. The GitHub key is read from memory and stays out of the command
      line:
  ```bash
  set -a; . /dev/shm/lmarch.env; set +a
  H='!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f'
  git -c credential.helper= -c credential.helper="$H" clone https://github.com/xieyiheng00-gif/model_setting.git
  git -c credential.helper= -c credential.helper="$H" clone https://github.com/xieyiheng00-gif/llm_training.git
  unset WANDB_API_KEY HF_TOKEN GITHUB_TOKEN; export LMARCH_SECRETS_FILE=/dev/shm/lmarch.env; cd model_setting
  ```
- [ ] B4. Install (PyTorch comes with the machine image). flash-attn comes as a prebuilt wheel, about 30 s
      instead of a 25-minute build:
  ```bash
  pip install -r requirements.txt nvitop flash-linear-attention
  python -c "import torch, sys; print(torch.__version__, torch.version.cuda, sys.version[:6])"   # expect 2.12.x 13.0 3.12
  pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl#sha256=1c0a88bbccf34378a24b580dc55b2e6b852ced1486510ba03bd309ef36a018e5"
  python -c "import flash_attn; print(flash_attn.__version__)"      # 2.8.3
  ```
  If the second line prints other versions, the wheel doesn't fit: pick the matching one or build from
  source (see "GPU kernels").
- [ ] B5. Check the GPUs:
  - `nvidia-smi` lists 8 × H100;
  - `nvidia-smi topo -m` shows `NV18` between every pair (NVLink).
- [ ] B6. Download the data (16.5 GB):
  ```bash
  python scripts/with_secrets.py -- hf download xie1231/llm-10b-packed --repo-type dataset --local-dir /data/llm/packed
  ```

### C. Checks before spending the hours (about 20 min)
- [ ] C1. Tests: `pytest -q tests`. This includes the checks that the fused kernels match the plain code on
      the GPU.
- [ ] C2. Speed check on one GPU (about 10 min):
  ```bash
  python scripts/bench_step.py --config configs/h100x8/trunk.yaml --arch kda_full --ab
  python scripts/bench_step.py --config configs/h100x8/s2_16k.yaml --arch kda_full --ab
  ```
  Expect:
  - `attn_backend: flash_varlen` and `kda_backend: fla`;
  - `compile_ops: on (self-check ok …)` and `fused_ce: on (self-check ok …)`;
  - matching first-step losses, and a speed-up. Note the tokens/s and MFU.
- [ ] C3. Pre-flight on all 8 GPUs (5–8 min: 40 steps, 2 checkpoint uploads):
  ```bash
  torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/h100x8/trunk.yaml --arch kda_full --run_name kda_full_preflight --set train.max_steps=40 schedule.warmup_steps=5 checkpoint.interval=20 checkpoint.hf_every=20 eval.interval=0 eval.final_full=false log.wandb_group=preflight
  ```
  - [ ] The start-up line shows `gpu_perf_counters: on (8 of 8 GPUs)`.
  - [ ] The W&B run `kda_full_preflight` has points in `train/loss`, `train/mfu` and the `gpu` section
        (`gpu/sm_active_mean`, `gpu/tensor_active_mean`, `gpu/dram_active_mean`, `gpu/power_w_mean`).
  - [ ] `python -m lmarch.train.hf_upload --status runs/kda_full_preflight` lists 2 checkpoints as
        `uploaded`.
  - [ ] Optional: afterwards delete `kda_full_preflight/` from the Hub repo and the run from W&B.

### D. Start the run (1 min)
- [ ] D1. `bash scripts/start_training.sh`
  - It checks the keys, then runs `ARCHS=kda_full bash scripts/run_all.sh h100x8` inside a new, detached
    tmux session called `train`.
  - Window `train` holds the run, which is also logged to `logs/train_<date>.log`. Window `gpu` shows
    nvitop.
  - Other architectures: `ARCHS="dense kda_full" bash scripts/start_training.sh`.
- [ ] D2. `tmux attach -t train`. Within a few minutes you should see:
  - the W&B link;
  - the runtime line: `flash_varlen`, `fla`, `compile_ops on`, `fused_ce on`, `gpu_perf_counters on`;
  - then step lines with tokens/s and MFU.
- [ ] D3. Detach with **Ctrl-b d**, then close SSH and your computer. The run keeps going on the server.

### E. While it runs: where to look
| what | where |
|---|---|
| loss curve | W&B project `model_setting`, run `kda_full_trunk_h100x8` (then `…_s2_16k_h100x8`, `…_s2_4k_h100x8`): `train/loss` |
| speed | `train/tokens_per_sec`, `train/mfu` |
| GPU hardware, per step (mean over the 8 GPUs, same x-axis as the loss) | W&B section `gpu`: `sm_active_mean`, `tensor_active_mean`, `dram_active_mean`, `sm_occupancy_mean`, `power_w_mean`, `util_mean`, `util_min_gpu`, `sm_mhz_min`, `mem_gb_max`, `temp_c_max`, `power_capped_frac` |
| each GPU separately (every 5 s) | the run's **System** tab in W&B: GPU utilization, SM active, tensor-pipe active, DRAM active, power, clocks, NVLink |
| the machine itself | `ssh gpu`, `tmux attach -t train`, **Ctrl-b n** for the nvitop window |
| problems | W&B alerts (e-mail or Slack): a crash, rollback or failed upload, and each stage's end |

How to read the GPU counters (all in %):
- `sm_active` is the share of the SMs that have work.
- `tensor_active` is the share of cycles the tensor cores (the matmul units) are busy. It is closest to MFU.
- `dram_active` is the share of the peak memory bandwidth in use.
- High `sm_active` with low `tensor_active` and high `dram_active` means memory-bound kernels: the fused
  kernels in "Training speed" target exactly that.
- `sm_active` dropping to near 0 marks pauses: checkpoints, evals, data waits.

### F. After training, before stopping the machine
- [ ] F1. The log shows all three stages finished: `tail -n 30 logs/train_*.log`.
- [ ] F2. Every milestone and final checkpoint is on the Hub:
      `python -m lmarch.train.hf_upload --status runs/kda_full_trunk_h100x8`, and the same for
      `…_s2_16k_h100x8` and `…_s2_4k_h100x8`.
- [ ] F3. `run_all.sh` wrote `reports/h100x8_trunk`, `reports/h100x8_stage2` and `reports/h100x8_gpu`.
      Copy them to your computer with `scp -r gpu:<repo path>/reports .`. The metrics and the GPU telemetry
      are also on the Hub.
- [ ] F4. Stop the machine. The key file in `/dev/shm` disappears with it; revoke any key you created only
      for this machine.

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
- **128K vocabulary.** The LM head and cross-entropy run in 2,048-token chunks, so the full logits never
  exist. Training steps use the fused version (`train.fused_ce`): each chunk's gradient is computed in the
  forward pass, so nothing is recomputed. Evals use the checkpointed chunked version.
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

### GPU kernels (H100)

Two optional packages decide the speed of a real run. Both are picked up automatically, and the
start-up line (and `meta.json` → `runtime`) says which path each run uses:

| package | used for | without it |
|---|---|---|
| `flash-attn` | F layers and the DSA dense warm-up with document masking (`attn_backend: flash_varlen`) | SDPA with a (B,T,T) mask (`sdpa_mask`): a start-up WARNING, and a note in the smoke summary |
| `flash-linear-attention` (fla) | KDA (`kda_backend: fla`, numerically self-checked at start-up) | the PyTorch chunked kernel |

The 2026-09-26 H100 speed test ran on `sdpa_mask` and got 12.7% MFU for dense 4K
(`results/h100_smoke_2026-09-26`). The official flash-attn release has no wheel for torch 2.12 + CUDA 13,
and building it from source took ~25 min. A prebuilt wheel installs in about 30 s instead:

```bash
python -c "import torch, sys; print(torch.__version__, torch.version.cuda, sys.version[:6])"   # expect 2.12.x 13.0 3.12
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl#sha256=1c0a88bbccf34378a24b580dc55b2e6b852ced1486510ba03bd309ef36a018e5"
python -c "import flash_attn; print(flash_attn.__version__)"   # 2.8.3; a mismatched build fails to import here
```

- **Match all three versions.** The wheel only fits torch 2.12, CUDA 13.0 and Python 3.12, which the first
  line prints. For another combination, take the file named `flash_attn-2.8.3+cu<CUDA>torch<torch>-cp<python>-…`
  from the [release page](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases).
- **Third-party build.** The wheel comes from mjun0812/flash-attention-prebuild-wheels, not from the
  flash-attn authors, and it runs inside the training process. The `#sha256=` pins the exact file that
  release published, and pip refuses anything else.
- **Fallback:** build from source, in tmux:
  ```bash
  pip install ninja packaging
  MAX_JOBS=10 pip install flash-attn --no-build-isolation   # ~25 min; MAX_JOBS bounds RAM use during the build
  ```

Install it once per machine or image, and before the real runs. `--set model.attn_backend=flash_varlen`
makes a run fail at start-up instead of falling back.

### Training speed (MFU)

The 1×H100 speed test reached 12.7% MFU (dense) and 8.4% (kda_full). Most of the time was not in the
matmuls. It went into memory-bound work:
- masked attention without flash-attn;
- the fp32 logits of the 128K-vocabulary output layer, computed twice (forward and recompute);
- chains of small elementwise ops (fp32 RMSNorm, SwiGLU, the KDA short conv, gates and L2 norms), each reading
  and writing the whole activation.

The H100 configs enable four exact speed-ups. Each is checked against the plain code at start-up and falls back
to it by itself; the start-up line and `meta.json` → `runtime` show what is in use.

| speed-up | setting | effect |
|---|---|---|
| flash-attn for document-masked full attention | automatic when installed | F layers skip cross-document blocks |
| fused LM head + cross-entropy | `train.fused_ce: true` | the gradient is computed during the forward pass: no fp32 logits kept, no recompute |
| compiled elementwise kernels | `train.compile_ops: true` (H100 configs) | `torch.compile` fuses RMSNorm, the residual add, SwiGLU, RoPE, the KDA conv, gates, L2 norm and output norm, and the loss softmax (`lmarch/model/fused.py`) |
| activation checkpointing only where needed | `train.activation_checkpointing: auto` (16K stage) | dense and KDA skip the recompute at 16K (they fit: the same activation memory as a 4 × 4K micro-batch); DSA and CSA keep it |

Notes:
- The first steps of a run take 1–3 min longer while the kernels compile.
- `--set train.compile_ops=false` (or the environment variable `LMARCH_NO_COMPILE=1`) and
  `--set train.fused_ce=false` turn the speed-ups off.
- If a 16K run without recompute ever runs out of memory, the supervisor restarts it with activation
  checkpointing on.

**Batch size.** The global batch (tokens per optimizer step) is part of the training recipe; changing it
changes the results. The micro-batch (tokens per forward pass, 16K per GPU) can change freely, but it already
fills the matmuls. The memory-bound work costs the same per token at any batch size, so a bigger micro-batch
helps only a few percent. Try it with `--set train.micro_batch_size=8` below.

**What to expect.**
- **Dense:** mostly matmuls, so the speed-ups should bring it close to the 25–35% range.
- **kda_full:** about half of its step is inside the nine KDA layers. The fla recurrence kernel does little
  counted work (MFU counts only the layers' projections) and is bound by memory and latency. Expect roughly
  1.5× (to about 12–15% MFU) until the KDA kernel itself gets faster.
- Measure it before a run (about 5 minutes on one GPU):
  ```bash
  python scripts/bench_step.py --config configs/h100x8/trunk.yaml --arch kda_full --ab
  python scripts/bench_step.py --config configs/h100x8/s2_16k.yaml --arch kda_full --ab
  python scripts/bench_step.py --config configs/h100x8/trunk.yaml --arch kda_full --profile
  ```
  `--ab` compares the plain code with the speed-ups, and checks that the first-step loss matches.
  `--profile` splits the GPU time by kernel family: matmul, attention, KDA, compiled ops, eager elementwise,
  loss, optimizer. That shows where the rest of the time goes.

## What is saved

Metrics go to `runs/<run>/metrics/*.jsonl` (append-only; readers keep the last record per step) and, live,
to W&B (next section),
checkpoints to `checkpoints/`, NaN forensics to `crash_reports/`, plus `meta.json`, `config.yaml` and
`heartbeat.json`.

- **Every step:** `loss` (global-token-weighted), `lr`, `grad_norm`, **`loss_nonfinite` /
  `grad_nonfinite`**, `step_time`, `tokens_per_sec`, `peak_mem_gb`, `skipped/action/reasons`, plus:
  - `mfu` (share of the GPUs' BF16 peak, see "GPU monitoring") and `gpu` (telemetry summary since the
    previous step);
  - `loss_by_source`, `loss_by_row_type`, `valid_tokens`, `long_token_frac`, `mean_segment_len`;
  - `data_index`, `data_batch`, `data_epoch`, indexer KL per layer and type.
- **Every 100 steps**, per layer and **grouped by layer type**:
  - activation RMS at each block output, per-group grad norms (before clipping), and update/weight
    ratios with a health label and alerts;
  - the update/weight ratio `‖ΔW‖/‖W‖` is taken over the weight **matrices** of a group. 1-D
    parameters (norm gains, KDA `A_log`/`dt_bias`, CSA biases/sink) are reported separately as
    `vec_ratio`, because their large norms would otherwise dominate the ratio;
  - ratio alerts start at `diag.alert_start_step` (1000). Right after warm-up the weights are still
    near their init, and ratios of ~1e-2 (the embedding most of all) are normal;
  - F/S/C: max logit and entropy per head;
  - S/C: indexer KL, attention-mass recall, effective sparsity and density;
  - C: sink/window mass;
  - K: forget-gate, β and state-norm distributions;
  - LM-logit max/std/logsumexp (chunked).
  - The probe batch is a fixed mix of holdout rows.
- **Evals:** holdout loss overall, by group, by in-document position, cross-length, and dense vs
  sparse.

## API keys (`secrets.env`)

| key | needed for |
|---|---|
| `WANDB_API_KEY` | live W&B dashboards |
| `HF_TOKEN` | `hf download xie1231/llm-10b-packed` (private dataset) |
| `GITHUB_TOKEN` | cloning or pushing this private repo from a GPU machine |

The keys live in **`secrets.env`** at the repo root. It is gitignored; `secrets.env.example` is the
committed template. On a shared machine, point `LMARCH_SECRETS_FILE` at a file in your home directory
instead. Fill it in with hidden input (nothing is echoed):
```bash
python scripts/setup_secrets.py            # or edit secrets.env yourself
python scripts/setup_secrets.py --verify   # prints only the account each key belongs to
```

`scripts/train.py` and `scripts/supervise.py` load the file automatically. For other commands, use
the wrapper:
```bash
python scripts/with_secrets.py -- hf download xie1231/llm-10b-packed --repo-type dataset --local-dir /data/llm/packed
python scripts/with_secrets.py -- wandb sync runs/<run>/wandb/offline-run-*
python scripts/with_secrets.py -- git push
```

**Safety, including on remote GPU nodes:**
- Keys go into process environments only, never onto command lines, so they don't show in `ps` or
  shell history.
- On Linux the file is forced to mode 600.
- Every log line, traceback, W&B alert and wrapped command output is filtered, so a key shows up as
  `***`.
- Keys never enter configs, checkpoints or W&B configs.
- `.claude/settings.json` blocks Claude Code from reading or printing the file. The programs load it;
  the assistant never sees the values.

**First clone on a new GPU box** (nothing exists there yet):
```bash
read -rs GITHUB_TOKEN && export GITHUB_TOKEN        # paste, hidden
git -c credential.helper= -c credential.helper='!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
    clone https://github.com/xieyiheng00-gif/model_setting.git
cd model_setting && python scripts/setup_secrets.py && unset GITHUB_TOKEN
```
Or copy your local file over an encrypted channel with `scp secrets.env gpu:~/model_setting/`
(then `chmod 600`). Prefer separate, narrowly scoped keys per machine: an HF read token, and a GitHub
fine-grained token limited to this repo. That way you can revoke one without touching the others.

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
(no package, auth, network, API changes) switch W&B off; they never stop training. The run records
them as a `wandb` event with `mode: failed` and the error, and `scripts/smoke.py` fails on them.

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

## GPU monitoring

**Utilization % alone is misleading.** NVML's "GPU utilization" (what `nvidia-smi` and `nvitop` show) only
measures how much of the time some kernel was running. It says nothing about how much of the GPU that kernel
uses: the 1×H100 speed test ran at 99.7% utilization and 12.7% MFU. So the trainer records three things.
Checklist step E says where to watch them during a run:

| what | where | how often |
|---|---|---|
| `mfu`: model FLOPs per second divided by the GPUs' BF16 peak (the speed report's convention) and `tokens_per_sec` | step record, W&B `train/mfu`, and the console line | every step |
| **every GPU**, read through NVML by rank 0: utilization, memory, **power**, temperature, SM clock, throttle reasons, and the hardware counters **SM active, SM occupancy, tensor-core active, DRAM active** (NVML GPM, Hopper and newer) | `metrics/gpu.jsonl` (also uploaded to the Hub with each checkpoint); per-step means in W&B `gpu/*` on the `trainer/step` axis, like the loss | every 1 s (`log.gpu_interval_sec`) |
| W&B's own system metrics, including **`smActive` and `pipeTensorActive`** (how busy the SMs and tensor cores really are), NVLink/PCIe traffic, CPU, RAM and disk | the run's System tab in W&B | every 5 s (`log.wandb_stats_interval_sec`; W&B's default is 15 s) |

Notes:
- The peak is taken from the GPU's name (H100 SXM: 989.4 TFLOPS dense BF16). Set `log.peak_tflops` for
  other GPUs.
- For S, C and K layers only their projections are counted, so their MFU is a lower bound.
- The monitor needs `nvidia-ml-py` (in `requirements.txt`). Without it the run logs a warning and trains
  normally. The start-up line shows `gpu_perf_counters: on (8 of 8 GPUs)` when the hardware counters work.

**Live view on the machine:** `scripts/start_training.sh` opens `nvitop` in the tmux window `gpu` (per-GPU
history, per-process memory; `nvidia-smi` if nvitop is missing). The hardware counters are only in W&B and
`gpu.jsonl`.

**After training** (or during it):
```bash
python scripts/gpu_report.py runs/kda_full_trunk_h100x8 runs/kda_full_s2_16k_h100x8 runs/kda_full_s2_4k_h100x8 --out reports/gpu
```
This writes `gpu_summary.md` / `.json` with one row per run:
- mean and 5th-percentile utilization, and minutes below 50%;
- the slowest GPU;
- power, SM clock, and the share of time power-capped or thermally throttled;
- peak memory and temperature;
- median tokens/s and MFU.

It also writes `<run>_gpu.png`: a utilization heatmap (GPU × time), then power, SM clock and MFU over time,
with checkpoint saves, evals and restarts marked. Finally it prints the longest low-utilization stretches
and what was happening during each (start-up, a checkpoint save, an eval, a restart, or nothing known,
which points at data loading or host overhead).

## Crash handling

The trainer skips non-finite steps and writes a forensics report naming the first non-finite module
and its layer type. Every loss or grad-norm spike logs a `guard` event with the gradient norms
(before clipping) per layer type and the five largest per layer. The step record gets `grad_culprit`
(e.g. `L05.mixer(kda) 83.2`), and the console line shows it. Repeated failures trigger a **rollback** to the last healthy checkpoint of the
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

## Checkpoints and Hugging Face uploads

A checkpoint is the full training state: fp32 weights, both Adam moments, trainer and guard state, and
RNG. That is about 2.2 GB at full size. Saves are atomic (a `COMPLETE` marker is written last).

| setting (`configs/base.yaml`) | meaning |
|---|---|
| `checkpoint.interval: 500` | a checkpoint every 500 steps |
| `checkpoint.keep_last: 3` | the newest 3 regular checkpoints stay on disk; older ones are deleted |
| `checkpoint.milestone_interval: 2500` | every 2,500 steps: a milestone, never deleted |
| `save_final: true` | `step_<last>_final` at the end of every stage, never deleted |

For one architecture on the full plan (times at the modelled 8×H100 speed of kda_full):

| stage | checkpoints saved | on disk at the end | uploaded to the Hub |
|---|---|---|---|
| trunk (steps 0 → 13,351) | 27: 500, 1,000 … 13,000 + final (one every ~8 min) | 9 (~20 GB) | 6: 2,500 … 12,500 + final (~13 GB, one every ~40 min) |
| s2_16k (13,351 → 18,770) | 12: 13,500 … 18,500 + final (every ~11 min) | 6 (~13 GB) | 3: 15,000, 17,500 + final (~7 GB) |
| s2_4k (13,351 → 18,770) | 12 (every ~8 min) | 6 (~13 GB) | 3 (~7 GB) |

**Uploads.** `checkpoint.hf_repo` is set in `configs/h100x8/*` and `configs/h100x1/*` to the private repo
`xie1231/model_setting-checkpoints`.
- **What goes up:** the milestones and the final checkpoint (`checkpoint.hf_every` changes that; it must
  be a multiple of 500). Each upload also takes a snapshot of the run's `meta.json`, `config.yaml`,
  `wandb_run_id.txt` and `metrics/*.jsonl`. The layout in the repo mirrors `runs/<run>/`.
- **How:** each upload runs in its own background process, so training never waits for it.
  - The upload continues if the supervisor restarts the trainer. A restarted trainer re-queues anything
    that is not marked uploaded.
  - A checkpoint that is still uploading is not deleted by the keep-3 cleanup.
  - At the normal end of a stage, the trainer waits up to `hf_wait_min` (60 min) for the uploads.
- **Where to look:**
  - `runs/<run>/hf_upload.log`;
  - `hf_upload` events in `metrics/events.jsonl`;
  - a W&B alert if an upload fails;
  - `python -m lmarch.train.hf_upload --status runs/<run>`.
- **Token:** `HF_TOKEN` needs write access to the checkpoint repo as well as read access to the dataset.

**Restoring on a new machine** (the rented box died): download the run into `runs/`, then start the same
training command again. It resumes from the newest downloaded checkpoint and continues the same W&B run.

```bash
python scripts/with_secrets.py -- hf download xie1231/model_setting-checkpoints --include "kda_full_trunk_h100x8/*" --local-dir runs
```

## Layout

```
lmarch/config.py        hyperparameters (dataclasses), YAML inherit (lists allowed), --set overrides
lmarch/data/packed.py   packed-data reader: trunk/stage-2 streams, labels, rank split, holdout, prefetch
lmarch/data/synthetic.py  synthetic data with the real schema (tests)
lmarch/model/           common (DocInfo, norms), fused (compilable elementwise ops), attention (F, S), indexer, kda, csa, model (fused CE)
lmarch/train/           trainer, guard, monitor, gpu_monitor, checkpoint (+ branch loading), hf_upload, logger, optim, dist
scripts/                train, supervise, smoke, make_smoke_data, analyze_runs, gpu_report, bench_step, run_all.{sh,ps1}
configs/                base, stage/{trunk,s2_16k,s2_4k}, {h100x8,h100x1,rtx3060,smoke,test}/*
```
