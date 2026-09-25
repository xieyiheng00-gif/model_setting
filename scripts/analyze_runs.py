"""Compare runs: summary table + plots + flattened CSVs.

  python scripts/analyze_runs.py runs/*_trunk_h100x8 --out reports/trunk            # stage 1
  python scripts/analyze_runs.py runs/*_s2_16k_h100x8 runs/*_s2_4k_h100x8 --out reports/stage2
  python scripts/analyze_runs.py runs_smoke/* --out reports/smoke

Log records are append-only; after a resume/rollback some steps appear twice. The LAST record per step
wins (it belongs to the trajectory that was continued); superseded ones are counted as "orphaned".
Requires pandas + matplotlib.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_jsonl(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # torn last line after a hard kill
    return out


def last_per_step(recs: list[dict]) -> tuple[list[dict], int]:
    by = {}
    for r in recs:
        by[r["step"]] = r
    return [by[s] for s in sorted(by)], len(recs) - len(by)


def flatten(d, prefix="", sep="."):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        elif isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float)):
            out[key] = v
    return out


def load_run(run_dir: Path) -> dict:
    import pandas as pd
    m = run_dir / "metrics"
    steps, orphan = last_per_step(read_jsonl(m / "train_steps.jsonl"))
    evals, _ = last_per_step(read_jsonl(m / "eval.jsonl"))
    diags, _ = last_per_step(read_jsonl(m / "diagnostics.jsonl"))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8")) if (run_dir / "meta.json").exists() else {}
    events = read_jsonl(m / "events.jsonl")
    sdf = pd.DataFrame([flatten(r) for r in steps])
    edf = pd.DataFrame([flatten(r) for r in evals])
    # diagnostics: by-type table (type, metric, stat) and per-layer table
    bt_rows, layer_rows = [], []
    for r in diags:
        for t, metrics in r.get("by_type", {}).items():
            for metric, stats in metrics.items():
                bt_rows.append({"step": r["step"], "type": t, "metric": metric, **stats})
        for L in r.get("layers", []):
            row = {"step": r["step"], "layer": L["idx"], "type": L["type"]}
            row.update(flatten({k: v for k, v in L.items() if k not in ("idx", "type")}))
            layer_rows.append(row)
        for k, v in r.get("global", {}).items():
            bt_rows.append({"step": r["step"], "type": "global", "metric": k, "mean": v, "min": v, "max": v, "n": 1})
    return {"name": run_dir.name, "dir": run_dir, "meta": meta, "steps": sdf, "eval": edf,
            "by_type": pd.DataFrame(bt_rows), "layers": pd.DataFrame(layer_rows), "events": events,
            "orphaned_step_records": orphan}


def summarize(run: dict) -> dict:
    s, e, ev = run["steps"], run["eval"], run["events"]
    row = {"run": run["name"], "arch": run["meta"].get("arch"), "stage": run["meta"].get("stage"),
           "pattern": run["meta"].get("pattern"),
           "params_M": (run["meta"].get("params", {}).get("total", float("nan")) / 1e6),
           "non_emb_M": (run["meta"].get("params", {}).get("non_embedding", float("nan")) / 1e6)}
    if len(s):
        row.update({
            "steps": int(s["step"].max()) + 1,
            "tokens_B": float(s["tokens_seen"].max()) / 1e9,
            "train_loss_last50": float(s["loss"].tail(50).mean()),
            "tok_per_s_median": float(s["tokens_per_sec"].median()),
            "tok_per_s_per_gpu_median": float(s["tokens_per_sec_per_gpu"].median()),
            "peak_mem_gb_max": float(s["peak_mem_gb"].max()),
            "nonfinite_steps": int(((s["loss_nonfinite"] + s["grad_nonfinite"]) > 0).sum()),
            "skipped_steps": int(s["skipped"].sum()),
            "spikes": int(s["spike"].sum()) if "spike" in s else 0,
            "grad_norm_median": float(s["grad_norm"].median()),
        })
    if len(e):
        full = e[e["full"] == 1] if "full" in e else e.iloc[0:0]
        last = full.iloc[-1] if len(full) else e.iloc[-1]      # prefer the final FULL holdout eval
        row["val_loss_final"] = float(last["val_loss"])
        row["val_final_is_full"] = bool(len(full))
        row["val_loss_best_periodic"] = float(e["val_loss"].min())
        for c in e.columns:
            if c.startswith(("sparse_minus_dense", "val_loss@", "val_by_group", "val_by_pos")) and c in last:
                row[c] = float(last[c])
    row["rollbacks"] = sum(1 for x in ev if x.get("kind") == "rollback")
    row["ratio_alerts"] = sum(1 for x in ev if x.get("kind") == "ratio_alert")
    row["restarts"] = sum(1 for x in ev if x.get("kind") == "resume")
    row["orphaned_step_records"] = run["orphaned_step_records"]
    return row


def plot_all(runs: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def fig_lines(fname, title, series, ylog=False, ylabel=""):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, x, y in series:
            if len(x):
                ax.plot(x, y, label=label, linewidth=1.3)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        if ylog:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)

    def ema(v, a=0.95):
        o, m = [], None
        for x in v:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                o.append(m)
                continue
            m = x if m is None else a * m + (1 - a) * x
            o.append(m)
        return o

    for src in ("fineweb_edu", "finepdfs_edu", "books", "code", "math"):
        col = f"loss_by_source.{src}"
        ser = [(r["name"], r["steps"]["step"], ema(r["steps"][col].tolist())) for r in runs
               if len(r["steps"]) and col in r["steps"]]
        if ser:
            fig_lines(f"train_loss_{src}.png", f"train loss on {src} rows (EMA)", ser)
    fig_lines("train_loss.png", "train loss (EMA 0.95)",
              [(r["name"], r["steps"]["step"], ema(r["steps"]["loss"].tolist())) for r in runs if len(r["steps"])])
    fig_lines("val_loss.png", "validation loss",
              [(r["name"], r["eval"]["step"], r["eval"]["val_loss"]) for r in runs if len(r["eval"])])
    for col, title, ylog in (("grad_norm", "global grad norm (pre-clip)", True), ("tokens_per_sec", "tokens/sec", False),
                             ("peak_mem_gb", "peak memory (GB, max over ranks)", False), ("lr", "learning rate", False)):
        fig_lines(f"{col}.png", title, [(r["name"], r["steps"]["step"], r["steps"][col])
                                        for r in runs if len(r["steps"]) and col in r["steps"]], ylog)

    def by_type_series(metric, stat="mean"):
        series = []
        for r in runs:
            bt = r["by_type"]
            if not len(bt):
                continue
            sub = bt[bt["metric"] == metric]
            for t, g in sub.groupby("type"):
                series.append((f"{r['name']}:{t}", g["step"], g[stat]))
        return series

    for metric, title, ylog, stat in (
        ("update.mixer.ratio", "mixer update/weight ratio (target ~1e-3)", True, "mean"),
        ("update.ffn.ratio", "FFN update/weight ratio", True, "mean"),
        ("grad_norm.mixer", "mixer grad norm", True, "mean"),
        ("act.act_rms", "block-output activation RMS (max over layers)", True, "max"),
        ("mixer.max_logit.max", "max pre-softmax logit", False, "max"),
        ("mixer.norm_entropy.mean", "normalised attention entropy", False, "mean"),
        ("mixer.recall_mean", "indexer attention-mass recall", False, "mean"),
        ("mixer.indexer_kl_dense", "indexer KL vs dense attention (probe)", True, "mean"),
        ("mixer.effective_sparsity", "effective sparsity", False, "mean"),
        ("mixer.forget_gate.mean", "KDA forget gate mean", False, "mean"),
        ("mixer.forget_gate.min", "KDA forget gate min", False, "min"),
        ("mixer.beta.mean", "KDA beta mean", False, "mean"),
        ("mixer.state_norm_max", "KDA state norm at sequence end (max)", True, "max"),
        ("mixer.sink_mass.mean", "CSA sink mass", False, "mean"),
    ):
        s = by_type_series(metric, stat)
        if s:
            fig_lines("diag_" + metric.replace(".", "_") + ".png", title, s, ylog)

    # final holdout loss per group
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / max(len(runs), 1)
    groups = sorted({c.split(".", 1)[1] for r in runs for c in r["eval"].columns if c.startswith("val_by_group.")})
    for j, r in enumerate(runs):
        e = r["eval"]
        if not len(e) or not groups:
            continue
        last = e.iloc[-1]
        ax.bar([i + j * width for i in range(len(groups))],
               [last.get(f"val_by_group.{g}", float("nan")) for g in groups], width, label=r["name"])
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(groups))])
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
    ax.set_title("final holdout loss by source/row_type")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "val_by_group.png", dpi=130)
    plt.close(fig)

    # loss vs position (last eval)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in runs:
        e = r["eval"]
        if not len(e):
            continue
        cols = [c for c in e.columns if c.startswith("val_by_pos.")]
        if cols:
            ax.plot([c.split(".", 1)[1] for c in cols], [e.iloc[-1][c] for c in cols], marker="o", label=r["name"])
    ax.set_title("final holdout loss by position inside the document")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "val_loss_by_position.png", dpi=130)
    plt.close(fig)


def main() -> int:
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="reports/latest")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(Path(p)) for p in a.runs if (Path(p) / "metrics").exists()]
    if not runs:
        print("no runs with metrics/ found")
        return 1
    summary = pd.DataFrame([summarize(r) for r in runs])
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print(summary.to_string(index=False))
    for r in runs:
        d = out / r["name"]
        d.mkdir(exist_ok=True)
        r["steps"].to_csv(d / "train_steps.csv", index=False)
        r["eval"].to_csv(d / "eval.csv", index=False)
        r["by_type"].to_csv(d / "diag_by_type.csv", index=False)
        r["layers"].to_csv(d / "diag_layers.csv", index=False)
    try:
        plot_all(runs, out)
    except ImportError:
        print("matplotlib not installed: skipped plots")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
