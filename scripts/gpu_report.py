"""GPU utilization report from the trainer's telemetry, for analysis after (or during) training.

    python scripts/gpu_report.py runs/kda_full_trunk_h100x8 runs/kda_full_s2_16k_h100x8 --out reports/gpu

Reads runs/<run>/metrics/gpu.jsonl (every GPU sampled through NVML, 1 s; lmarch/train/gpu_monitor.py),
train_steps.jsonl (tokens/s, MFU), events.jsonl (checkpoints, restarts, rollbacks), eval.jsonl and
diagnostics.jsonl, and writes to --out:
  gpu_summary.md / gpu_summary.json   one row per run: utilization (mean, 5th percentile, time below 50%), the
                                      slowest GPU, power, SM clock, throttling, peak memory and temperature,
                                      tokens/s and MFU
  <run>_gpu.png                       utilization heatmap (GPU x time), then power, SM clock and MFU over time,
                                      with checkpoint saves, evals and restarts marked
It also prints the longest low-utilization stretches and what the trainer was doing at the time.
Utilization only means that some kernel was running: read it together with MFU and power.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# reference light palette (dataviz skill): chart chrome, one series hue, status "critical", sequential blue ramp
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES, CRITICAL = "#2a78d6", "#d03b3b"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]
LOW_UTIL = 50.0                          # % mean utilization below which a stretch counts as "low"
POWER_BITS, THERMAL_BITS = 0x4 | 0x80, 0x8 | 0x20 | 0x40


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass                         # the line being written while we read
    return out


def _matrix(samples: list[dict], key: str, n_gpu: int) -> np.ndarray:
    a = np.full((len(samples), n_gpu), np.nan)
    for r, s in enumerate(samples):
        for g in s["gpus"]:
            v = g.get(key)
            if v is not None and g["i"] < n_gpu:
                a[r, g["i"]] = v
    return a


def activities(steps: list[dict], events: list[dict], evals: list[dict], diags: list[dict]) -> list[tuple]:
    """(kind, start, end) wall-clock intervals of what the trainer did besides training steps."""
    t_of = {s["step"]: s["time"] for s in steps if "time" in s}
    out = []
    for e in events:
        if e.get("kind") == "checkpoint" and "time" in e:
            out.append(("checkpoint save", e["time"] - float(e.get("seconds") or 0), e["time"]))
        elif e.get("kind") in ("resume", "rollback", "start", "branch_init") and "time" in e:
            out.append(("restart" if e["kind"] in ("resume", "rollback") else "start", e["time"], e["time"]))
    for recs, kind, dur_key in ((evals, "eval", "eval_time"), (diags, "diagnostics", "diag_time")):
        for r in recs:
            d = float(r.get(dur_key) or 0)
            if "time" in r:              # newer runs: wall-clock end of the eval / diagnostics
                out.append((kind, r["time"] - d, r["time"]))
            else:                        # older runs: it followed the step record before it
                t0 = t_of.get(r["step"] - 1 if kind == "eval" else r["step"])
                if t0 is not None:
                    out.append((kind, t0, t0 + d))
    return out


def analyze(run_dir: Path) -> dict | None:
    m = run_dir / "metrics"
    samples = [s for s in read_jsonl(m / "gpu.jsonl") if s.get("gpus")]
    if not samples:
        return None
    steps = [s for s in read_jsonl(m / "train_steps.jsonl") if "time" in s]
    acts = activities(steps, read_jsonl(m / "events.jsonl"), read_jsonl(m / "eval.jsonl"),
                      read_jsonl(m / "diagnostics.jsonl"))
    n_gpu = max(len(s["gpus"]) for s in samples)
    t = np.array([s["time"] for s in samples], dtype=float)
    dt = np.diff(t, append=t[-1] + (np.median(np.diff(t)) if len(t) > 1 else 1.0))
    util, power = _matrix(samples, "util", n_gpu), _matrix(samples, "power_w", n_gpu)
    sm, mem, temp = _matrix(samples, "sm_mhz", n_gpu), _matrix(samples, "mem_gb", n_gpu), _matrix(samples, "temp_c", n_gpu)
    thr = _matrix(samples, "throttle", n_gpu)
    with np.errstate(all="ignore"):
        um = np.nanmean(util, axis=1)
        per_gpu = np.nanmean(util, axis=0)
    valid_thr = thr[~np.isnan(thr)].astype(np.int64)
    toks = [s["tokens_per_sec"] for s in steps[3:] if s.get("tokens_per_sec")]
    mfus = [s["mfu"] for s in steps[3:] if s.get("mfu") is not None]
    slow = int(np.nanargmin(per_gpu))
    summary = {
        "run": run_dir.name, "hours": round((t[-1] - t[0]) / 3600, 2), "gpus": n_gpu,
        "util_mean": round(float(np.nanmean(util)), 1), "util_p5": round(float(np.nanpercentile(um, 5)), 1),
        "low_util_min": round(float(dt[um < LOW_UTIL].sum() / 60), 1),
        "slowest_gpu": slow, "slowest_gpu_util": round(float(per_gpu[slow]), 1),
        "fastest_gpu_util": round(float(np.nanmax(per_gpu)), 1),
        "power_w_mean": round(float(np.nanmean(power)), 0),
        "sm_mhz_median": round(float(np.nanmedian(sm)), 0), "sm_mhz_p5": round(float(np.nanpercentile(sm, 5)), 0),
        "power_capped_pct": round(100 * float(np.mean(valid_thr & POWER_BITS > 0)), 1) if valid_thr.size else None,
        "thermal_pct": round(100 * float(np.mean(valid_thr & THERMAL_BITS > 0)), 1) if valid_thr.size else None,
        "mem_gb_peak": round(float(np.nanmax(mem)), 1), "temp_c_peak": round(float(np.nanmax(temp)), 0),
        "tok_s_median": round(float(np.median(toks))) if toks else None,
        "mfu_median_pct": round(100 * float(np.median(mfus)), 1) if mfus else None,
    }
    return {"summary": summary, "t": t, "dt": dt, "util": util, "um": um, "power": power, "sm": sm,
            "steps": steps, "acts": acts, "stretches": low_stretches(t, dt, um, steps, acts)}


def low_stretches(t, dt, um, steps, acts, top: int = 5) -> list[dict]:
    """The longest runs of consecutive samples with mean utilization below LOW_UTIL, with what overlapped them."""
    low = um < LOW_UTIL
    runs, i = [], 0
    while i < len(low):
        if low[i]:
            j = i
            while j + 1 < len(low) and low[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    first = steps[0]["time"] - steps[0].get("step_time", 0) if steps else None
    last = steps[-1]["time"] if steps else None
    out = []
    for i, j in sorted(runs, key=lambda r: t[r[1]] + dt[r[1]] - t[r[0]], reverse=True)[:top]:
        a, b = t[i], t[j] + dt[j]
        what = sorted({k for k, s, e in acts if s <= b + 2 and e >= a - 2})
        if first is not None and a < first:
            what.append("start-up before the first step")
        if last is not None and b > last + 2:
            what.append("after the last step (final eval, save, upload wait)")
        step = next((s["step"] for s in steps if s["time"] >= a), None)
        out.append({"start_h": round((a - t[0]) / 3600, 3), "minutes": round((b - a) / 60, 2), "step": step,
                    "util_mean": round(float(np.nanmean(um[i:j + 1])), 1),
                    "during": ", ".join(what) or "no checkpoint, eval or diagnostics overlaps"})
    return out


def _binned(x, y, edges, reduce):
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    out = np.full(len(edges) - 1, np.nan)
    for b in np.unique(idx):
        v = y[idx == b]
        v = v[~np.isnan(v)]
        if v.size:
            out[b] = reduce(v)
    return out


def plot(r: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    t, s = r["t"], r["summary"]
    t0, dur = t[0], max(t[-1] - t[0], 1.0)
    nb = int(min(720, max(2, dur / max(float(np.median(r["dt"])), 1e-3))))
    edges = np.linspace(t0, t0 + dur, nb + 1)
    xh = (edges[:-1] + edges[1:]) / 2
    xh = (xh - t0) / 3600
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(4, 2, width_ratios=[60, 1], height_ratios=[1.5, 1, 1, 1], hspace=0.55, wspace=0.02)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[k, 0], sharex=axes[0]) for k in (1, 2, 3)]
    cax = fig.add_subplot(gs[0, 1])

    # 1) utilization, GPU x time
    n_gpu = r["util"].shape[1]
    heat = np.stack([_binned(t, r["util"][:, g], edges, np.mean) for g in range(n_gpu)])
    cmap = LinearSegmentedColormap.from_list("util", BLUES).with_extremes(bad=SURFACE)   # no samples: surface
    im = axes[0].imshow(np.ma.masked_invalid(heat), aspect="auto", cmap=cmap, vmin=0, vmax=100,
                        interpolation="nearest", extent=[0, dur / 3600, n_gpu - 0.5, -0.5])
    axes[0].set_yticks(range(n_gpu), [f"GPU {g}" for g in range(n_gpu)])
    axes[0].set_title("GPU utilization, % of time a kernel was running (darker = busier)")
    cb = fig.colorbar(im, cax=cax, ticks=[0, 50, 100])
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)

    # 2) power, 3) SM clock: mean over GPUs with the min-max band across GPUs
    for ax, key, title in ((axes[1], "power", "Power per GPU, W (line: mean over GPUs, band: lowest to highest GPU)"),
                           (axes[2], "sm", "SM clock, MHz (line: mean over GPUs, band: lowest to highest GPU)")):
        a = r[key]
        with np.errstate(all="ignore"):
            mean = _binned(t, np.nanmean(a, axis=1), edges, np.mean)
            lo = _binned(t, np.nanmin(a, axis=1), edges, np.min)
            hi = _binned(t, np.nanmax(a, axis=1), edges, np.max)
        ax.fill_between(xh, lo, hi, color=SERIES, alpha=0.12, linewidth=0)
        ax.plot(xh, mean, color=SERIES, linewidth=1.1, solid_capstyle="round")
        ax.set_title(title)

    # 4) MFU (or tokens/s) from the step records
    st = r["steps"]
    mfu = np.array([x.get("mfu") if x.get("mfu") is not None else np.nan for x in st], dtype=float)
    ts = np.array([x["time"] for x in st], dtype=float)
    if st and not np.all(np.isnan(mfu)):
        y, title = 100 * mfu, "MFU, % of the GPUs' BF16 peak (from the trainer; median per bin)"
    else:
        y = np.array([x.get("tokens_per_sec", np.nan) for x in st], dtype=float) / 1e3
        title = "Throughput, thousand tokens/s (from the trainer; median per bin)"
    if st:
        axes[3].plot(xh, _binned(ts, y, edges, np.median), color=SERIES, linewidth=1.1, solid_capstyle="round")
    axes[3].set_title(title)
    axes[3].set_xlabel("hours since the first sample")

    # what the trainer was doing: neutral hairlines, restarts in the status colour (with a legend label)
    styles = {"checkpoint save": (AXIS, 0.7), "eval": (INK2, 0.7), "restart": (CRITICAL, 1.2)}
    seen = set()
    for kind, a, _ in r["acts"]:
        if kind in styles and t0 <= a <= t0 + dur:
            c, lw = styles[kind]
            for ax in axes:
                ax.axvline((a - t0) / 3600, color=c, linewidth=lw, zorder=0 if kind != "restart" else 3)
            seen.add(kind)
    for ax in axes[1:]:
        ax.set_ylim(bottom=0)                        # magnitudes: a zero baseline, so noise does not look like swings
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    if seen:
        handles = [Line2D([], [], color=styles[k][0], linewidth=1.5, label=k) for k in styles if k in seen]
        fig.legend(handles=handles, loc="upper right", frameon=False, ncol=len(handles), labelcolor=INK2,
                   bbox_to_anchor=(0.98, 0.985))
    mfu_txt = f" · MFU median {s['mfu_median_pct']}%" if s["mfu_median_pct"] is not None else ""
    fig.suptitle(f"{s['run']}: {s['gpus']} GPUs, {s['hours']} h · mean utilization {s['util_mean']}% · "
                 f"{s['low_util_min']} min below {LOW_UTIL:.0f}%{mfu_txt}", x=0.07, ha="left", y=0.99,
                 color=INK, fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


COLUMNS = [("run", "run"), ("hours", "hours"), ("gpus", "GPUs"), ("util_mean", "util mean %"),
           ("util_p5", "util p5 %"), ("low_util_min", f"min < {LOW_UTIL:.0f}% util"),
           ("slowest_gpu_util", "slowest GPU util %"), ("power_w_mean", "power W"),
           ("sm_mhz_median", "SM MHz median"), ("sm_mhz_p5", "SM MHz p5"), ("power_capped_pct", "power-capped %"),
           ("thermal_pct", "thermal %"), ("mem_gb_peak", "mem peak GB"), ("temp_c_peak", "temp peak °C"),
           ("tok_s_median", "tok/s median"), ("mfu_median_pct", "MFU median %")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (runs/<run>)")
    ap.add_argument("--out", default="reports/gpu")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rd in map(Path, a.runs):
        r = analyze(rd)
        if r is None:
            print(f"{rd.name}: no metrics/gpu.jsonl (runs from before the GPU monitor, or not on CUDA)")
            continue
        s = r["summary"]
        rows.append(dict(s, low_util_stretches=r["stretches"]))
        print(f"\n{s['run']}: {s['gpus']} GPUs, {s['hours']} h, mean utilization {s['util_mean']}%, "
              f"{s['low_util_min']} min below {LOW_UTIL:.0f}%, slowest GPU {s['slowest_gpu']} "
              f"({s['slowest_gpu_util']}%), MFU median {s['mfu_median_pct']}%")
        for x in r["stretches"]:
            print(f"  {x['minutes']:6.2f} min at {x['start_h']:.2f} h (step {x['step']}), util {x['util_mean']}%: "
                  f"{x['during']}")
        if not a.no_plots:
            plot(r, out / f"{s['run']}_gpu.png")
    if not rows:
        return 1
    md = ["| " + " | ".join(h for _, h in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    md += ["| " + " | ".join("" if row.get(k) is None else str(row.get(k)) for k, _ in COLUMNS) + " |" for row in rows]
    (out / "gpu_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "gpu_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'gpu_summary.md'}, {out / 'gpu_summary.json'}"
          + ("" if a.no_plots else f" and {len(rows)} plot(s)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
