# H100 smoke test + GPU speed record — 2026-09-26

Machine: 1 × NVIDIA H100 80GB HBM3 (SXM5, 132 SMs, 700 W, 1980 MHz max), Xeon Platinum 8468 (20 vCPU), 235 GB RAM.
Software: torch 2.12.0+cu130, triton 3.7.0, fla 0.5.2 (KDA kernel, passed self-check), flash-attn 2.8.3 (built here,
installed after the speed runs), wandb 0.30.0. Repos: model_setting @ 0f00790, llm_training @ 0fa2dd0.

## 1. Smoke test (configs/smoke, d_model 256, real data slice) — PASSED

`python scripts/smoke.py --set log.wandb_mode=offline` (then `wandb sync`), 01:06 → 01:22 (16 min).

| arch | trunk (0→40) | s2_16k (40→60) | s2_4k (40→60) | full-holdout val loss trunk → 16K / 4K |
|---|---|---|---|---|
| dense    | OK | OK | OK | 8.511 → 8.392 / 8.389 |
| dsa      | OK | OK | OK | 8.571 → 8.526 / 8.450 |
| kda_full | OK | OK | OK | 8.589 → 8.487 / 8.467 |
| kda_dsa  | OK | OK | OK | 8.592 → 8.489 / 8.477 |
| csa      | OK | OK | OK | 8.488 → 8.410 / 8.410 |

All automatic checks passed: exit 0 everywhere, metrics/diagnostics/eval written, both branches initialised from
the trunk's final checkpoint at the same step and read the same stage-2 batches, LR decayed to ~0, no NaN/Inf, no
skipped steps, no rollbacks, per-type diagnostics present for every layer type.
W&B: all 15 runs synced to https://wandb.ai/xieyiheng00-university-of-wisconsin-madison/model_setting
(groups trunk_smoke / s2_16k_smoke / s2_4k_smoke). The `--faults` rehearsal was not run.

## 2. Raw GPU speed vs the H100 SXM5 datasheet  (gpu_bench.py → gpu_bench.json)

| test | this GPU | datasheet peak | % of peak | typical healthy H100 SXM |
|---|---|---|---|---|
| BF16 GEMM 4096 / 8192 / 16384 | 777 / 763 / 727 TFLOPS | 989 | 73–79 % | ~650–800 |
| BF16 GEMM 8192, sustained 60 s | 680 avg (587–726) TFLOPS | 989 | 69 % | power-capped, same |
| FP16 GEMM | 636–746 TFLOPS | 989 | 64–75 % | similar to BF16 |
| TF32 GEMM | 333–387 TFLOPS | 495 | 67–78 % | |
| FP8 GEMM (`_scaled_mm`) | 1219–1418 TFLOPS | 1979 | 62–72 % | |
| HBM3 read / copy | 3.16 / 3.01 TB/s | 3.35 | 94 / 90 % | ~2.9–3.1 |
| PCIe host→GPU / GPU→host (pinned) | 51.5 / 55.2 GB/s | 64 (Gen5 x16) | 80–86 % | ~50–55 |

Sustained load: 698 W average (at the 700 W cap), SM clock settles at ~1360 MHz under pure GEMM with throttle
reason 0x4 = SW power cap (normal for H100 SXM on dense GEMMs), max 66 °C — no thermal throttling.
During model training the card stayed at the full 1980 MHz, 500–666 W, 50–62 °C.
**Verdict: a healthy, standard H100 SXM5. Nothing is degraded.**

## 3. Training speed, FULL-SIZE model on 1 H100  (speed_report.py → speed_summary.md/json)

Real per-GPU micro-batches of configs/h100x8 (4 × 4K; 16K = 1 × 16K + activation checkpointing), 131K tokens per
optimizer step, smoke data slice, no evals in the timed window. Median over steady steps (first 5 and last excluded).
MFU = model FLOPs (6·N_matmul incl. the tied 128K×768 LM head + causal attention of F layers) / 989 TFLOPS.
Backend: SDPA with a (B,T,T) document mask (flash-attn was not yet installed); KDA on the fla Triton kernel.

| arch | stage | tok/s | vs dense 4K | s/step | peak mem GB | model TFLOPS | MFU % | GPU util % | power W |
|---|---|---|---|---|---|---|---|---|---|
| dense    | 4K  | 95,042 | 1.00x | 1.38 | 15.9 | 126.1 | 12.7 | 99.7 | 596 |
| dsa      | 4K  | 30,391 | 0.32x | 4.31 | 31.2 | 34.0* | 3.4* | 99.9 | 504 |
| kda_full | 4K  | 71,351 | 0.75x | 1.84 | 23.0 | 83.4* | 8.4* | 96.6 | 527 |
| kda_dsa  | 4K  | 50,855 | 0.54x | 2.58 | 26.8 | 56.8* | 5.7* | 97.8 | 506 |
| dense    | 16K | 48,049 | 0.51x | 2.73 | 7.2  | 96.4  | 9.7  | 99.9 | 666 |
| dsa      | 16K | 5,834  | 0.06x | 22.47 | 15.3 | 6.5* | 0.7* | 100 | 498 |
| kda_full | 16K | 51,806 | 0.55x | 2.53 | 7.2  | 69.3* | 7.0* | 96.4 | 535 |

\* S/C/K mixer cores are not counted in model FLOPs, so these MFUs are lower bounds.
Not measured (stopped on request): kda_dsa 16K, csa 4K/16K, the flash-attn rerun and the kernel profile.

**Reading:** the GPU is ~100 % busy at full clock, so it is not starved by the CPU or the data loader — it is
running slow kernels. 12.7 % MFU for the dense baseline is low for an H100 (well-tuned training of models this size
typically reaches roughly 30–45 %). Likely causes, not yet profiled: SDPA with a dense boolean document mask
(memory-efficient kernel, no skipping of masked blocks), fp32 cross-entropy over the 128K vocab with recompute,
no torch.compile. DSA/CSA use dense-mask kernels for exact top-k (README caveat); at 16K DSA builds 16K×16K fp32
indexer scores per layer, hence 22 s/step.

**What this means for the real plan (8 × H100, assuming linear scaling, evals excluded):**

| arch | trunk 7.0B tok (4K) | s2_16k 2.84B tok |
|---|---|---|
| dense    | ~2.6 h | ~2.1 h |
| dsa      | ~8.0 h | **~17 h** |
| kda_full | ~3.4 h | ~1.9 h |
| kda_dsa  | ~4.8 h | not measured |

## 4. Issues found on the way

1. **W&B broke on current wandb (fixed locally, uncommitted):** `lmarch/train/wandb_sink.py` called
   `wandb.util.generate_id()`, which wandb 0.30 moved to `wandb.sdk.lib.runid`. Training continued but W&B was
   silently disabled. Fix = import fallback (git diff in model_setting). `requirements.txt` allows `wandb>=0.17`,
   so a fresh 8×H100 box would hit the same bug until this is committed.
2. **Pushed repo is older than your README:** no `secrets.env` in `.gitignore`, no `.claude/settings.json`,
   no `scripts/setup_secrets.py` / `with_secrets.py`. Here `secrets.env` is excluded via `.git/info/exclude`.
   Push your local version before cloning on other GPU boxes.
3. The first W&B key (42 chars) was rejected by W&B; the new 86-char key works.
4. flash-attn has no prebuilt wheel for torch 2.12 + CUDA 13; building it from source took ~25 min (10 jobs).
   It is now installed on this box, so future dense/kda_full runs here pick `flash_varlen` automatically.

## Files in this folder

| path | contents |
|---|---|
| `bench/` | `gpu_bench.py` (GEMM / HBM / PCIe / sustained-power microbenchmark), its `.json` result and log |
| `speed/` | `run_speed.sh`, `speed_report.py`, `speed_summary.{md,json}`, `gpu_trace.csv` (1 Hz nvidia-smi: SM clock, util, power, temp) |
| `speed/configs/` | full-size speed-test configs (inherit `configs/h100x1/*`; paths are the absolute `/workspace/...` ones used on the box) |
| `speed/runs/<run>/` | `meta.json`, `config.yaml`, `metrics/*.jsonl` of each speed run |
| `speed/logs/` | console log of each speed run |
| `smoke/smoke_all.log` | full console output of `scripts/smoke.py` (ends with the smoke summary) |
| `smoke/runs/<run>/` | `meta.json`, `config.yaml`, `metrics/*.jsonl`, supervisor events, exit status of each smoke run |
| `smoke/report/` | `scripts/analyze_runs.py` output for the smoke runs (per-run tables + diagnostic plots) |

Not included: checkpoints (runs_smoke 17 GB, runs_speed 15 GB), the data slice, W&B offline binaries (the smoke runs
are already synced to W&B), the flash-attn build log.
