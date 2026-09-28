"""Training speed of one architecture on this GPU: synthetic packed batches through the real model, optimizer and
kernels (no data, no trainer), with an optional A/B of the speed-ups and a profile of where the GPU time goes.

    python scripts/bench_step.py --config configs/h100x8/trunk.yaml --arch kda_full
    python scripts/bench_step.py --config configs/h100x8/trunk.yaml --arch kda_full --ab
    python scripts/bench_step.py --config configs/h100x8/s2_16k.yaml --arch kda_full --ab
    python scripts/bench_step.py --config configs/h100x8/trunk.yaml --arch kda_full --profile
    python scripts/bench_step.py ... --set train.micro_batch_size=8

One optimizer step = --accum micro-batches of train.micro_batch_size x train.seq_len tokens (4 = the per-GPU shape
of the 8xH100 configs) with realistic document lengths, then the fused AdamW step. Reported: tokens/s, MFU (the
trainer's definition), step time, peak memory and the kernels each variant used. Eval shapes are run once too, so
torch.compile has compiled them before a real run needs them.
  --ab       baseline (fused_ce and compile_ops off, activation checkpointing as before) vs the config as given,
             and the first-step loss of both (the speed-ups are exact up to rounding)
  --profile  torch.profiler over 3 steps: GPU time by kernel family and the top kernels; --trace FILE also writes a
             Chrome trace (open in https://ui.perfetto.dev)
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from lmarch.config import activation_checkpointing, build_config  # noqa: E402
from lmarch.model import RunFlags, build_model  # noqa: E402
from lmarch.model import fused  # noqa: E402
from lmarch.train.gpu_monitor import peak_flops  # noqa: E402
from lmarch.train.optim import build_optimizer  # noqa: E402


def synthetic_batch(B: int, T: int, vocab: int, mean_doc: float, gen: torch.Generator, device):
    """Random tokens in documents of exponentially distributed length; labels = next token, -100 at document ends."""
    ids = torch.randint(0, vocab, (B, T), generator=gen)
    pos = torch.empty(B, T, dtype=torch.long)
    for b in range(B):
        t = 0
        while t < T:
            n = int(min(T - t, max(1.0, torch.empty(1).exponential_(1.0 / mean_doc, generator=gen).item())))
            pos[b, t:t + n] = torch.arange(n)
            t += n
    labels = torch.full((B, T), -100, dtype=torch.long)
    same_doc = pos[:, 1:] != 0
    labels[:, :-1] = torch.where(same_doc, ids[:, 1:], torch.full_like(ids[:, 1:], -100))
    return ids.to(device), labels.to(device), pos.to(device)


def run_variant(name: str, cfg, a, device) -> dict:
    fused.disable()
    torch.manual_seed(0)
    model = build_model(cfg.model).to(device).train()
    model.grad_checkpointing = activation_checkpointing(cfg)
    model.loss_chunk_tokens = cfg.train.loss_chunk_tokens
    bf16 = cfg.train.dtype == "bf16"
    info = model.configure_runtime(device, cfg.data.doc_mask, bf16, fused_ce=cfg.train.fused_ce,
                                   compile_ops=cfg.train.compile_ops)
    info["activation_checkpointing"] = model.grad_checkpointing
    opt = build_optimizer(model, cfg.optim, cfg.model.indexer.lr_mult, device)
    B, T = cfg.train.micro_batch_size, cfg.train.seq_len
    mean_doc = a.mean_doc or (8000.0 if T > 4096 else 1000.0)
    gen = torch.Generator().manual_seed(1)
    batches = [synthetic_batch(B, T, cfg.model.vocab_size, mean_doc, gen, device) for _ in range(a.accum)]
    flags = RunFlags(sparse=True, indexer_loss=bool(model.indexer_layers), diag=False)
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=bf16)

    def step() -> float:
        losses = []
        for x, y, pos in batches:
            with torch.autocast(**amp):
                out = model(x, y, flags, pos if cfg.data.doc_mask else None)
            loss = out["loss_sum"] / out["n_valid"].clamp(min=1) / a.accum
            if "aux_loss" in out:
                loss = loss + out["aux_loss"] / a.accum
            loss.backward()
            losses.append((out["loss_sum"] / out["n_valid"].clamp(min=1)).detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return torch.stack(losses).mean().item()

    t_c = time.perf_counter()
    first_loss = step()                                  # also compiles (compile_ops) - not timed
    compile_s = time.perf_counter() - t_c
    for _ in range(max(0, a.warmup - 1)):
        step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(a.steps):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    dt = statistics.median(times)
    tokens = a.accum * B * T
    peak = peak_flops(torch.cuda.get_device_name(device), cfg.log.peak_tflops)
    res = {"variant": name, "tok_s": tokens / dt, "step_s": dt, "peak_gb": torch.cuda.max_memory_allocated(device) / 2**30,
           "mfu": model.flops_per_token(T) * tokens / dt / peak if peak else None, "first_loss": first_loss,
           "first_step_s": compile_s, "info": info}
    if a.eval:                                             # eval shapes: compiled (and timed) before a real run needs them
        model.eval()
        ev = cfg.eval.micro_batch_size or B
        for L in [T] + [int(x) for x in cfg.eval.extra_seq_lens]:
            x, y, pos = synthetic_batch(ev, L, cfg.model.vocab_size, mean_doc, gen, device)
            with torch.no_grad(), torch.autocast(**amp):
                model(x, y, RunFlags(sparse=True), pos if cfg.data.doc_mask else None)
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                model(x, y, RunFlags(sparse=True), pos if cfg.data.doc_mask else None)
                torch.cuda.synchronize(device)
            res[f"eval_tok_s@{L}"] = ev * L / (time.perf_counter() - t0)
        model.train()
    if a.profile:
        profile(step, a.trace, device)
    del model, opt, batches
    fused.disable()
    torch.cuda.empty_cache()
    return res


FAMILIES = [   # (family, regex on the lower-cased kernel name); first match wins
    ("communication (NCCL)", r"nccl"),
    ("memcpy / memset", r"memcpy|memset"),
    ("attention (flash-attn / SDPA)", r"flash|fmha|efficient_attention|attention"),
    ("optimizer (fused AdamW)", r"adam|multi_tensor"),
    ("compiled fused ops (inductor)", r"^triton_"),
    ("KDA (fla Triton kernels)", r"chunk_|solve_tril|recompute_w|wy_|kkt|kda|gated_delta|l2norm|fla"),
    ("matmul (cuBLAS / CUTLASS)", r"gemm|xmma|nvjet|cutlass|cublas|gemv|splitk|sm90_|sm80_|ampere_|hopper"),
    ("loss (log-softmax / CE)", r"softmax|cross_entropy|nll_loss"),
    ("eager elementwise / reductions", r"elementwise|reduce|vectorized|unrolled|index|scatter|gather|cat|copy|fill|"
                                       r"embedding|sort|cumsum|where|pow|rsqrt|mul|add"),
]


def profile(step, trace, device) -> None:
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprofile
    n = 3
    with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(n):
            step()
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - t0
    fam: dict = {}
    kernels = []
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", None)
        if us is None:
            us = getattr(e, "self_cuda_time_total", 0)
        if not us:
            continue
        name = e.key
        f = next((f for f, rx in FAMILIES if re.search(rx, name.lower())), "other")
        fam[f] = fam.get(f, 0.0) + us
        kernels.append((us, f, name))
    total = sum(fam.values())
    print(f"\n  GPU kernel time {total / 1e3 / n:.1f} ms/step of {wall * 1e3 / n:.1f} ms wall "
          f"(GPU busy {100 * total / 1e6 / wall:.0f}%)")
    print(f"  {'kernel family':34s} {'ms/step':>8s} {'share':>6s}")
    for f, us in sorted(fam.items(), key=lambda kv: -kv[1]):
        print(f"  {f:34s} {us / 1e3 / n:8.1f} {100 * us / total:5.1f}%")
    print("  top kernels:")
    for us, f, name in sorted(kernels, reverse=True)[:15]:
        print(f"    {us / 1e3 / n:7.2f} ms  {f[:22]:22s} {name[:95]}")
    if trace:
        prof.export_chrome_trace(trace)
        print(f"  trace written to {trace}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step (8xH100 per-GPU shape: 4)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--mean_doc", type=float, default=0.0, help="mean document length (default 1000; 8000 above 4K)")
    ap.add_argument("--ab", action="store_true", help="also run the baseline without the speed-ups")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--trace", default="")
    ap.add_argument("--no-eval", dest="eval", action="store_false")
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 2
    device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = build_config(a.config, a.arch, overrides=a.set)
    variants = []
    if a.ab:
        old_ac = "true" if cfg.train.activation_checkpointing == "auto" else cfg.train.activation_checkpointing
        base = build_config(a.config, a.arch, overrides=a.set + ["train.fused_ce=false", "train.compile_ops=false",
                                                                 f"train.activation_checkpointing={old_ac}"])
        variants.append(("baseline", base))
    variants.append(("as configured", cfg))
    print(f"{torch.cuda.get_device_name(device)} | {a.arch} | {a.config} | {cfg.train.micro_batch_size} x "
          f"{cfg.train.seq_len} tokens x {a.accum} micro-steps per step")
    results = []
    for name, c in variants:
        print(f"\n== {name}", flush=True)
        r = run_variant(name, c, a, device)
        results.append(r)
        for k in ("attn_backend", "kda_backend", "compile_ops", "fused_ce", "activation_checkpointing"):
            if k in r["info"]:
                print(f"  {k}: {r['info'][k]}")
        mfu = f"{100 * r['mfu']:.1f}%" if r["mfu"] is not None else "n/a"
        evals = "  ".join(f"{k.split('@')[1]}: {v / 1e3:.0f}k tok/s" for k, v in r.items() if k.startswith("eval_tok_s"))
        print(f"  {r['tok_s'] / 1e3:.1f}k tok/s | MFU {mfu} | {1e3 * r['step_s']:.0f} ms/step | peak {r['peak_gb']:.1f} GB | "
              f"first step {r['first_step_s']:.0f}s (compile) | first-step loss {r['first_loss']:.5f}"
              + (f" | eval {evals}" if evals else ""))
    if len(results) == 2:
        b, o = results
        print(f"\nspeed-up {o['tok_s'] / b['tok_s']:.2f}x ({b['tok_s'] / 1e3:.1f}k -> {o['tok_s'] / 1e3:.1f}k tok/s); "
              f"first-step loss difference {abs(o['first_loss'] - b['first_loss']):.2e} (should be ~1e-3 or less)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
