"""Tiny synthetic dataset with the exact layout/schema of the packed data (tests, offline checks).

    write_synthetic_packed("tmp/packed", vocab_size=512)
creates trunk_4k/ (4096-token rows), stage2/ (32 x 16384 rows per step, row_type/long_tokens/batch),
holdout/ (7 source x row_type groups of 16384-token rows) + manifests, so the real reader
(lmarch/data/packed.py + data_prep.loader) runs unchanged. Tokens are random except EOS at document ends.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HOLDOUT_GROUPS = [("fineweb_edu", "long"), ("finepdfs_edu", "long"), ("books", "long"), ("code", "long"),
                  ("fineweb_edu", "short"), ("code", "short"), ("math", "short")]
SOURCES = ["fineweb_edu", "finepdfs_edu", "books", "code", "math"]


def _rows(rng, n: int, L: int, vocab: int, eos: int, long: np.ndarray):
    ids = rng.integers(1, vocab, size=(n, L)).astype(np.int32)
    ids[ids == eos] = eos - 1 if eos > 1 else eos + 1
    doc_lens = []
    for r in range(n):
        lens, left = [], L
        while left > 0:
            d = int(rng.integers(8192, 2 * 8192)) if long[r] else int(rng.geometric(1 / 300)) + 1
            d = min(d, left)
            lens.append(d)
            left -= d
        ends = np.cumsum(lens) - 1
        ids[r, ends] = eos
        doc_lens.append(np.array(lens, np.int32))
    return ids, doc_lens


def _table(ids, doc_lens, rng, source, row_type=None, batch=None):
    import pyarrow as pa
    n = ids.shape[0]

    def lst(arrs, typ):
        offs = np.concatenate([[0], np.cumsum([len(a) for a in arrs])]).astype(np.int32)
        return pa.ListArray.from_arrays(pa.array(offs), pa.array(np.concatenate(arrs), type=typ))

    cols = {"input_ids": lst(list(ids), pa.int32()), "doc_lens": lst(doc_lens, pa.int32()),
            "doc_keys": lst([rng.integers(0, 2**63, len(d), dtype=np.uint64) for d in doc_lens], pa.uint64()),
            "source": pa.array(source, type=pa.string())}
    if row_type is not None:
        cols["row_type"] = pa.array(row_type, type=pa.string())
        cols["long_tokens"] = pa.array([int(d[d >= 8192].sum()) for d in doc_lens], type=pa.int32())
    if batch is not None:
        cols["batch"] = pa.array(batch, type=pa.int32())
    return pa.table(cols)


def _write(table, out: Path, name: str, manifest: dict):
    import pyarrow.parquet as pq
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / name, row_group_size=1024, compression="zstd")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_synthetic_packed(out_dir: str | Path, vocab_size: int = 512, eos_id: int | None = None,
                           trunk_rows: int = 256, stage2_batches: int = 8, holdout_rows_per_group: int = 2,
                           seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    eos = vocab_size - 1 if eos_id is None else eos_id
    # trunk: 4K rows, random sources, no row_type
    ids, dl = _rows(rng, trunk_rows, 4096, vocab_size, eos, np.zeros(trunk_rows, bool))
    src = list(rng.choice(SOURCES, trunk_rows))
    _write(_table(ids, dl, rng, src), out / "trunk_4k", "trunk_4k-00000.parquet",
           {"seq_len": 4096, "rows": trunk_rows, "tokens": trunk_rows * 4096, "synthetic": True})
    # stage 2: 32 x 16K rows per step, ~70% long rows, stored in batch order
    n = stage2_batches * 32
    long = rng.random(n) < 0.7
    ids, dl = _rows(rng, n, 16384, vocab_size, eos, long)
    src = [str(rng.choice(["fineweb_edu", "finepdfs_edu", "books", "code"])) if lg else
           str(rng.choice(["fineweb_edu", "code", "math"])) for lg in long]
    rt = ["long" if lg else "short" for lg in long]
    _write(_table(ids, dl, rng, src, rt, np.repeat(np.arange(stage2_batches), 32)), out / "stage2",
           "stage2-00000.parquet", {"seq_len": 16384, "batch_rows": 32, "batches": stage2_batches,
                                    "tokens": n * 16384, "synthetic": True})
    # holdout: 7 groups, interleaved like the real file
    groups = [g for _ in range(holdout_rows_per_group) for g in HOLDOUT_GROUPS]
    long = np.array([t == "long" for _, t in groups])
    ids, dl = _rows(rng, len(groups), 16384, vocab_size, eos, long)
    _write(_table(ids, dl, rng, [s for s, _ in groups], [t for _, t in groups]), out / "holdout",
           "holdout-00000.parquet", {"seq_len": 16384, "rows": len(groups), "synthetic": True})
    return out
