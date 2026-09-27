"""Summarize the full-size H100 speed runs: steady-state tokens/sec, step time, memory, MFU, and the GPU's
power/clock/utilization during the timed steps (from gpu_trace.csv). Writes speed_summary.json + .md.

MFU convention (model FLOPs, recompute not counted):
  6 * N_matmul per token  (every Linear weight + the tied LM head V x d; embedding lookup = 0 FLOPs)
+ causal full attention in F layers: 6 * T * n_heads * head_dim per token per F layer.
S/C/K mixer cores (top-k attention, compressed attention, delta-rule recurrence) are not counted, so their
MFU is a lower bound -- and their dense-mask kernels are not representative anyway (README caveat).
"""
import csv, datetime as dt, json, statistics, sys
from pathlib import Path

ROOT = Path("/workspace/model_setting")
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from lmarch.config import build_config  # noqa: E402
from lmarch.model.model import build_model  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs_speed"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
PEAK_BF16 = 989.4e12          # H100 SXM5 datasheet, dense BF16
ARCHS = ["dense", "dsa", "kda_full", "kda_dsa", "csa"]


def flops_per_token(arch: str, cfg_path: Path, seq_len: int) -> dict:
    cfg = build_config(str(cfg_path), arch)
    with torch.device("meta"):
        model = build_model(cfg.model)
    # unique matmul weights: every Linear (a tied LM head shares the embedding tensor, count it once)
    ws = {id(m.weight): m.weight for m in model.modules() if isinstance(m, torch.nn.Linear)}
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding) and cfg.model.tie_embeddings:
            ws[id(m.weight)] = m.weight          # LM head = F.linear(h, embedding.weight)
    n_matmul = sum(w.numel() for w in ws.values())
    n_f = cfg.model.pattern().count("F")
    attn = 6 * seq_len * cfg.model.n_heads * cfg.model.head_dim * n_f
    return {"n_matmul": n_matmul, "matmul_flops": 6 * n_matmul, "attn_flops": attn,
            "total": 6 * n_matmul + attn, "pattern": cfg.model.pattern()}


def gpu_trace():
    p = HERE / "gpu_trace.csv"
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for r in csv.reader(f):
            try:
                t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                rows.append((t, float(r[1]), float(r[2]), float(r[3]), float(r[4])))
            except (ValueError, IndexError):
                continue
    return rows


def main():
    trace = gpu_trace()
    out = []
    for stage, cfg_name in (("trunk", "trunk.yaml"), ("s2_16k", "s2_16k.yaml")):
        for arch in ARCHS:
            rd = RUNS / f"{arch}_{stage}_h100speed"
            f = rd / "metrics" / "train_steps.jsonl"
            if not f.exists():
                continue
            recs = {}
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                    recs[r["step"]] = r
                except json.JSONDecodeError:
                    pass
            steps = [recs[k] for k in sorted(recs)]
            meta = json.loads((rd / "meta.json").read_text())
            # steady state: drop the first 5 steps of the run (allocator / autotune / step-0 diagnostics,
            # DSA+CSA dense warm-up) and the last step (diagnostics)
            steady = [s for s in steps[5:-1] if not s["skipped"]]
            if not steady:
                continue
            tps = statistics.median(s["tokens_per_sec"] for s in steady)
            st = statistics.median(s["step_time"] for s in steady)
            fl = flops_per_token(arch, HERE / "configs" / cfg_name, meta["seq_len"])
            t0, t1 = steady[0]["time"] - steady[0]["step_time"], steady[-1]["time"]
            win = [r for r in trace if t0 <= r[0] <= t1]
            avg = lambda i: round(sum(r[i] for r in win) / len(win), 1) if win else None
            out.append({
                "arch": arch, "stage": stage, "pattern": fl["pattern"], "seq_len": meta["seq_len"],
                "micro_batch": meta["micro_batch_size"], "grad_accum": meta["grad_accum"],
                "tokens_per_step": meta["tokens_per_step"], "attn_backend": meta["runtime"].get("attn_backend"),
                "kda_backend": meta["runtime"].get("kda_backend"),
                "params_total_M": round(meta["params"]["total"] / 1e6, 1),
                "steady_steps": len(steady), "step_time_s": round(st, 3), "tokens_per_sec": round(tps),
                "peak_mem_gb": round(max(s["peak_mem_gb"] for s in steady), 1),
                "gflop_per_token": round(fl["total"] / 1e9, 3),
                "achieved_tflops": round(tps * fl["total"] / 1e12, 1),
                "mfu_pct": round(100 * tps * fl["total"] / PEAK_BF16, 1),
                "gpu_power_w": avg(3), "gpu_sm_mhz": avg(1), "gpu_util_pct": avg(2), "gpu_temp_c": avg(4),
                "loss_first": round(steps[0]["loss"], 3), "loss_last": round(steps[-1]["loss"], 3),
            })
    (HERE / f"speed_summary{SUFFIX}.json").write_text(json.dumps(out, indent=2))
    dense4k = next((r["tokens_per_sec"] for r in out if r["arch"] == "dense" and r["stage"] == "trunk"), None)
    lines = ["| arch | stage | tok/s | vs dense 4K | step (s) | peak mem GB | TFLOPS | MFU % | power W | SM MHz | util % | backends |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        rel = f"{r['tokens_per_sec'] / dense4k:.2f}x" if dense4k else "-"
        be = r["attn_backend"] + (f", kda={r['kda_backend']}" if r.get("kda_backend") else "")
        lines.append(f"| {r['arch']} | {r['stage']} | {r['tokens_per_sec']:,} | {rel} | {r['step_time_s']} | "
                     f"{r['peak_mem_gb']} | {r['achieved_tflops']} | {r['mfu_pct']} | {r['gpu_power_w']} | "
                     f"{r['gpu_sm_mhz']} | {r['gpu_util_pct']} | {be} |")
    (HERE / f"speed_summary{SUFFIX}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
