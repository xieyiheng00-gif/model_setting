"""Build a small subset of the packed data (same folders, schema and manifests) for smoke tests.

  python scripts/make_smoke_data.py                                   # from D:/data/llm/packed or /data/llm/packed
  python scripts/make_smoke_data.py --src /data/llm/packed --dst data/packed_smoke

Default subset (~90 MB): trunk = first 2 row groups (2,048 x 4K rows = 8.4M tokens, 512 steps of 4 rows),
stage2 = first row group (32 complete steps x 32 x 16K rows, in training order), holdout = the first 8
rows of each of the 7 (source, row_type) groups. Only numpy + pyarrow are needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _default_src() -> str:
    for c in ("D:/data/llm/packed", "/data/llm/packed"):
        if Path(c, "trunk_4k").exists():
            return c
    return "D:/data/llm/packed"


def _first_row_groups(split_dir: Path, n: int):
    import pyarrow as pa
    import pyarrow.parquet as pq
    tables = []
    for path in sorted(split_dir.glob("*.parquet")):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            if len(tables) == n:
                return pa.concat_tables(tables)
            tables.append(pf.read_row_group(rg))
    return pa.concat_tables(tables)


def _write(table, out_dir: Path, name: str, manifest: dict) -> None:
    import pyarrow.parquet as pq
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / name, row_group_size=1024, compression="zstd")
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _counts(table, keys) -> list[dict]:
    cols = {k: table.column(k).to_pylist() for k in keys}
    n = table.num_rows
    seq = len(table.column("input_ids")[0])
    out: dict = {}
    for i in range(n):
        key = tuple(cols[k][i] for k in keys)
        out[key] = out.get(key, 0) + 1
    return [dict(zip(keys, k), rows=v, tokens=v * seq) for k, v in sorted(out.items())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=_default_src())
    ap.add_argument("--dst", default="data/packed_smoke")
    ap.add_argument("--trunk_row_groups", type=int, default=2)
    ap.add_argument("--stage2_row_groups", type=int, default=1)
    ap.add_argument("--holdout_rows_per_group", type=int, default=8)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if not (src / "trunk_4k").exists():
        print(f"{src} does not look like the packed data (no trunk_4k/)", file=sys.stderr)
        return 2
    t0 = time.time()

    trunk = _first_row_groups(src / "trunk_4k", a.trunk_row_groups)
    _write(trunk, dst / "trunk_4k", "trunk_4k-00000.parquet",
           {"seq_len": 4096, "rows": trunk.num_rows, "tokens": trunk.num_rows * 4096, "subset_of": str(src),
            "entries": _counts(trunk, ["source"])})

    s2 = _first_row_groups(src / "stage2", a.stage2_row_groups)
    batch = np.asarray(s2.column("batch").to_numpy())
    full = [b for b in np.unique(batch) if (batch == b).sum() == 32]
    keep = np.flatnonzero(np.isin(batch, full))
    s2 = s2.take(keep)
    _write(s2, dst / "stage2", "stage2-00000.parquet",
           {"seq_len": 16384, "batch_rows": 32, "batches": len(full), "tokens": s2.num_rows * 16384,
            "subset_of": str(src), "entries": _counts(s2, ["source", "row_type"])})

    import pyarrow.parquet as pq
    import pyarrow as pa
    hold = pa.concat_tables([pq.read_table(p) for p in sorted((src / "holdout").glob("*.parquet"))])
    src_col, rt_col = hold.column("source").to_pylist(), hold.column("row_type").to_pylist()
    seen: dict = {}
    idx = []
    for i, key in enumerate(zip(src_col, rt_col)):
        if seen.get(key, 0) < a.holdout_rows_per_group:
            seen[key] = seen.get(key, 0) + 1
            idx.append(i)
    hold = hold.take(np.array(idx))
    _write(hold, dst / "holdout", "holdout-00000.parquet",
           {"seq_len": 16384, "rows": hold.num_rows, "tokens": hold.num_rows * 16384, "subset_of": str(src),
            "entries": _counts(hold, ["source", "row_type"])})

    size = sum(p.stat().st_size for p in dst.rglob("*.parquet")) / 2**20
    print(f"wrote {dst}: trunk {trunk.num_rows} rows ({trunk.num_rows // 4} smoke steps of 4 rows), "
          f"stage2 {len(full)} steps, holdout {hold.num_rows} rows, {size:.0f} MB in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
