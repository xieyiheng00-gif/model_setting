"""Reader for the packed Llama-3 dataset built by llm_training/data_prep (see TRAINING_DATA.md).

    packed/trunk_4k/*.parquet   stage 1: 4K rows, already shuffled, read in file order
    packed/stage2/*.parquet     stage 2: 5,419 fixed steps x 32 rows x 16K (4K branch = same rows re-cut)
    packed/holdout/*.parquet    eval: 7 groups (source x row_type) x 305 rows x 16K

Everything here is numpy/pyarrow only (torch is imported by the Prefetcher), and uses the canonical
helpers from `data_prep.loader` (iter_stage2_batches, rank_rows, recut, doc_labels, position_ids).

Step semantics. The trainer asks for data index i = step - stage_start_step + data_skip:
  trunk:  rows [i*R, (i+1)*R) of the file-order row stream (R = trunk_rows_per_step), exactly the
          batches of TRAINING_DATA.md's `iter_trunk_batches`, but resuming skips whole row groups via
          parquet metadata instead of reading them.
  stage2: batch i of the stage-2 files (same tokens for the 16K and 4K branches).
Past the end of the split the stream wraps (epoch + 1), which only happens after rollbacks skipped data.

Per step, every rank computes the number of non-ignored labels of the WHOLE global batch, so the loss
can be averaged over the global batch (sum over GPUs and micro-batches / global count) without an
extra collective.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

SOURCES = ("fineweb_edu", "finepdfs_edu", "books", "code", "math", "other")
ROW_TYPES = ("long", "short", "trunk")
N_GROUPS = len(SOURCES) * len(ROW_TYPES)
IGNORE = -100


def group_name(g: int) -> str:
    return f"{SOURCES[g // len(ROW_TYPES)]}/{ROW_TYPES[g % len(ROW_TYPES)]}"


def import_data_prep(root: str = ""):
    """Import `data_prep.loader` from llm_training. Search order: already importable, `root`,
    $LMARCH_DATA_PREP, ../llm_training next to this repository."""
    try:
        import data_prep.loader as L  # type: ignore
        return L
    except ImportError:
        pass
    here = Path(__file__).resolve().parents[2]
    for c in (root, os.environ.get("LMARCH_DATA_PREP", ""), str(here.parent / "llm_training")):
        if c and (Path(c) / "data_prep" / "loader.py").exists():
            sys.path.insert(0, str(Path(c).resolve()))
            import data_prep.loader as L  # type: ignore
            return L
    raise ImportError("cannot import data_prep.loader: copy llm_training/data_prep next to this repo, set "
                      "data.data_prep_root, or export LMARCH_DATA_PREP=<folder containing data_prep/>")


# --------------------------------------------------------------------------------------
# row-level helpers
# --------------------------------------------------------------------------------------
def _read_rg(pf, rg: int, columns=None) -> dict:
    t = pf.read_row_group(rg, columns=columns)
    n = t.num_rows
    ids = t.column("input_ids").combine_chunks()
    offs = ids.offsets.to_numpy()
    out = {"input_ids": ids.values.to_numpy()[offs[0]:offs[-1]].reshape(n, -1)}
    dl = t.column("doc_lens").combine_chunks()
    o2, v2 = dl.offsets.to_numpy(), dl.values.to_numpy()
    out["doc_lens"] = [v2[o2[i]:o2[i + 1]] for i in range(n)]
    out["source"] = t.column("source").to_pylist()
    for name in ("row_type", "long_tokens"):
        if name in t.column_names:
            col = t.column(name)
            out[name] = col.to_pylist() if name == "row_type" else col.to_numpy()
    return out


def take_rows(rows: dict, idx) -> dict:
    out = {}
    for k, v in rows.items():
        if isinstance(v, np.ndarray):
            out[k] = v[idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx]
        else:
            out[k] = v
    return out


def recut_rows(batch: dict, seq_len: int) -> dict:
    """Cut every row into rows of seq_len (segments split at the new row edges). Same algorithm as
    data_prep.loader.recut, but also works for trunk rows (no row_type / long_tokens columns)."""
    n, old = batch["input_ids"].shape
    if old == seq_len:
        return batch
    k = old // seq_len
    assert k * seq_len == old, "seq_len must divide the row length"
    doc_lens = []
    for lens in batch["doc_lens"]:
        edges = np.union1d(np.concatenate([[0], np.cumsum(lens)]), np.arange(0, old + 1, seq_len))
        for j in range(k):
            e = edges[(edges >= j * seq_len) & (edges <= (j + 1) * seq_len)]
            doc_lens.append(np.diff(e).astype(np.int32))
    out = {"input_ids": batch["input_ids"].reshape(n * k, seq_len), "doc_lens": doc_lens,
           "source": [s for s in batch["source"] for _ in range(k)]}
    if "row_type" in batch:
        out["row_type"] = [t for t in batch["row_type"] for _ in range(k)]
    for key in ("batch", "epoch"):
        if key in batch:
            out[key] = batch[key]
    return out


def split_rows(batch: dict, rank: int, world: int, L) -> dict:
    """This rank's rows. Stage 2 / holdout: data_prep.rank_rows (same long/short mix on every GPU);
    trunk (no row_type): rows[rank::world] as in TRAINING_DATA.md."""
    if world == 1 and "row_type" not in batch:
        return batch
    if "row_type" in batch:
        return L.rank_rows(batch, rank, world)
    return take_rows(batch, list(range(rank, len(batch["source"]), world)))


def shifted_labels(input_ids: np.ndarray, doc_lens, L) -> np.ndarray:
    """Next-token targets: y[t] = doc_labels[t+1]; -100 at the last position and wherever the next
    token starts a new segment (never predict one document's first token from the previous one)."""
    lab = L.doc_labels(input_ids, doc_lens)
    y = np.full_like(lab, IGNORE)
    y[:, :-1] = lab[:, 1:]
    return y


def group_ids(batch: dict) -> np.ndarray:
    rt = batch.get("row_type")
    out = np.empty(len(batch["source"]), np.int64)
    for i, s in enumerate(batch["source"]):
        si = SOURCES.index(s) if s in SOURCES else len(SOURCES) - 1
        ri = ROW_TYPES.index(rt[i]) if rt is not None and rt[i] in ROW_TYPES else len(ROW_TYPES) - 1
        out[i] = si * len(ROW_TYPES) + ri
    return out


# --------------------------------------------------------------------------------------
# global-batch streams
# --------------------------------------------------------------------------------------
class TrunkStream:
    """Global batches of `rows_per_step` trunk rows in file order (identical to TRAINING_DATA.md's
    reference `iter_trunk_batches`); resume in O(1) using parquet row-group metadata."""

    def __init__(self, trunk_dir: str | Path, rows_per_step: int):
        import pyarrow.parquet as pq
        self.dir, self.R = Path(trunk_dir), rows_per_step
        self.groups = []                                   # (path, row_group, first_row, n_rows)
        start = 0
        for path in sorted(self.dir.glob("*.parquet")):
            md = pq.ParquetFile(path).metadata
            for rg in range(md.num_row_groups):
                n = md.row_group(rg).num_rows
                self.groups.append((path, rg, start, n))
                start += n
        if not self.groups:
            raise FileNotFoundError(f"no parquet files in {self.dir}")
        self.n_rows = start
        self.n_steps = self.n_rows // self.R               # the tail that doesn't fill a batch is dropped

    def iter_from(self, index: int) -> Iterator[dict]:
        import pyarrow.parquet as pq
        first = index * self.R
        g = max(i for i, (_, _, s, _) in enumerate(self.groups) if s <= first)
        skip = first - self.groups[g][2]
        buf: dict | None = None
        step = index
        files: dict = {}
        for path, rg, _, _ in self.groups[g:]:
            pf = files.setdefault(path, pq.ParquetFile(path))
            rows = _read_rg(pf, rg, columns=["input_ids", "doc_lens", "source"])
            if skip:
                rows = take_rows(rows, np.arange(skip, len(rows["source"])))
                skip = 0
            if buf is None:
                buf = rows
            else:
                buf = {"input_ids": np.concatenate([buf["input_ids"], rows["input_ids"]]),
                       "doc_lens": buf["doc_lens"] + rows["doc_lens"], "source": buf["source"] + rows["source"]}
            while len(buf["source"]) >= self.R and step < self.n_steps:
                out = take_rows(buf, np.arange(self.R))
                out["batch"] = step
                yield out
                buf = take_rows(buf, np.arange(self.R, len(buf["source"])))
                step += 1
            if step >= self.n_steps:
                return


class Stage2Stream:
    """Stage-2 global batches (always read as 16K rows; the 4K branch re-cuts them after optional
    subsampling, so both branches see the same tokens even in smoke mode)."""

    def __init__(self, stage2_dir: str | Path, L):
        self.dir, self.L = Path(stage2_dir), L
        man = Path(stage2_dir) / "manifest.json"
        if man.exists():
            with open(man, encoding="utf-8") as f:
                self.n_steps = int(json.load(f)["batches"])
        else:
            import pyarrow.parquet as pq
            last = sorted(self.dir.glob("stage2-*.parquet"))[-1]
            pf = pq.ParquetFile(last)
            col = pf.schema_arrow.get_field_index("batch")
            self.n_steps = int(pf.metadata.row_group(pf.num_row_groups - 1).column(col).statistics.max) + 1

    def iter_from(self, index: int) -> Iterator[dict]:
        for b in self.L.iter_stage2_batches(self.dir, view="16k", start_batch=index):
            if b["batch"] >= self.n_steps:
                return
            yield b


class PackedData:
    """The configured stage's data stream + step preparation for one rank."""

    def __init__(self, dcfg, seq_len: int, rank: int, world: int, micro: int, vocab_size: int):
        self.L = import_data_prep(dcfg.data_prep_root)
        self.cfg, self.seq_len, self.rank, self.world, self.micro = dcfg, seq_len, rank, world, micro
        self.vocab_size = vocab_size
        root = Path(dcfg.packed_dir)
        if dcfg.stage == "trunk":
            self.stream = TrunkStream(root / "trunk_4k", dcfg.trunk_rows_per_step)
            data_rows, row_len = dcfg.trunk_rows_per_step, 4096
        else:
            self.stream = Stage2Stream(root / "stage2", self.L)
            data_rows, row_len = 32, 16384
        if dcfg.max_rows_per_step and dcfg.max_rows_per_step < data_rows:
            if data_rows % dcfg.max_rows_per_step:
                raise ValueError(f"data.max_rows_per_step {dcfg.max_rows_per_step} must divide {data_rows}")
            data_rows = dcfg.max_rows_per_step
        self.rows_per_step = data_rows * (row_len // seq_len)
        self.tokens_per_step = self.rows_per_step * seq_len
        if self.rows_per_step % (micro * world):
            raise ValueError(f"{self.rows_per_step} rows per step can't be split into micro-batches of "
                             f"{micro} rows on {world} GPUs")
        self.accum = self.rows_per_step // (micro * world)
        self.n_steps = self.stream.n_steps

    def iter_steps(self, index: int) -> Iterator[tuple[int, dict]]:
        """(data_index, global batch) forever; wraps at the end of the split (epoch + 1)."""
        epoch, i = divmod(index, self.n_steps)
        while True:
            for b in self.stream.iter_from(i):
                b["epoch"] = epoch
                yield epoch * self.n_steps + b["batch"], b
            epoch, i = epoch + 1, 0

    def prepare(self, batch: dict) -> dict:
        """Global batch -> this rank's micro-batches (numpy) + global label count and step metadata."""
        cfg, L = self.cfg, self.L
        n = len(batch["source"])
        if cfg.max_rows_per_step and cfg.max_rows_per_step < n:
            stride = n // cfg.max_rows_per_step
            batch = L.rank_rows(batch, 0, stride) if "row_type" in batch else \
                take_rows(batch, np.arange(0, n, stride))
        long_frac = None
        if "long_tokens" in batch:
            long_frac = float(np.sum(batch["long_tokens"])) / batch["input_ids"].size
        batch = recut_rows(batch, self.seq_len)
        ids = batch["input_ids"]
        if cfg.check_token_ids and int(ids.max()) >= self.vocab_size:
            raise ValueError(f"token id {int(ids.max())} >= model.vocab_size {self.vocab_size}")
        global_valid = int((shifted_labels(ids, batch["doc_lens"], L) != IGNORE).sum())
        n_segments = sum(len(d) for d in batch["doc_lens"])
        mine = split_rows(batch, self.rank, self.world, L)
        micro = [self._arrays(take_rows(mine, np.arange(j, j + self.micro)))
                 for j in range(0, len(mine["source"]), self.micro)]
        return {"micro": micro, "global_valid": global_valid, "batch": batch.get("batch"),
                "epoch": batch.get("epoch", 0), "long_frac": long_frac,
                "mean_segment_len": ids.size / max(n_segments, 1)}

    def _arrays(self, rows: dict) -> dict:
        ids = rows["input_ids"]
        return {"ids": ids.astype(np.int64),
                "labels": shifted_labels(ids, rows["doc_lens"], self.L),
                "pos": self.L.position_ids(rows["doc_lens"], ids.shape[1]) if self.cfg.doc_mask else None,
                "group": group_ids(rows)}


# --------------------------------------------------------------------------------------
# holdout (evaluation + probe rows)
# --------------------------------------------------------------------------------------
class Holdout:
    """Holdout rows grouped by (source, row_type), in file order within each group."""

    def __init__(self, holdout_dir: str | Path, L):
        self.L = L
        rows = None
        for r in L.iter_rows(holdout_dir):
            r = {k: r[k] for k in ("input_ids", "doc_lens", "source", "row_type")}
            if rows is None:
                rows = r
            else:
                rows = {"input_ids": np.concatenate([rows["input_ids"], r["input_ids"]]),
                        "doc_lens": rows["doc_lens"] + r["doc_lens"], "source": rows["source"] + r["source"],
                        "row_type": rows["row_type"] + r["row_type"]}
        if rows is None:
            raise FileNotFoundError(f"no holdout parquet in {holdout_dir}")
        self.rows = rows
        self.by_group: dict[str, list[int]] = {}
        for i, (s, t) in enumerate(zip(rows["source"], rows["row_type"])):
            self.by_group.setdefault(f"{s}/{t}", []).append(i)

    def select(self, rows_per_group: int) -> dict:
        idx = sorted(i for g in self.by_group.values() for i in (g[:rows_per_group] if rows_per_group else g))
        return take_rows(self.rows, idx)

    def micro_batches(self, rows_per_group: int, seq_len: int, rank: int, world: int, micro: int,
                      doc_mask: bool) -> list[dict]:
        sel = recut_rows(self.select(rows_per_group), seq_len)
        mine = take_rows(sel, list(range(rank, len(sel["source"]), world)))
        out = []
        for j in range(0, len(mine["source"]), micro):
            r = take_rows(mine, np.arange(j, min(j + micro, len(mine["source"]))))
            out.append({"ids": r["input_ids"].astype(np.int64),
                        "labels": shifted_labels(r["input_ids"], r["doc_lens"], self.L),
                        "pos": self.L.position_ids(r["doc_lens"], seq_len) if doc_mask else None,
                        "group": group_ids(r)})
        return out

    def probe(self, n: int, seq_len: int, doc_mask: bool) -> dict:
        """n rows mixing the groups: the first seq_len-window of the first row of each group, round robin."""
        picks, k = [], 0
        groups = sorted(self.by_group)
        while len(picks) < n and k < 1000:
            g = self.by_group[groups[k % len(groups)]]
            j = k // len(groups)
            if j < len(g):
                picks.append(g[j])
            k += 1
        r = take_rows(self.rows, picks)
        r = recut_rows(r, seq_len)
        r = take_rows(r, [i * (16384 // seq_len) for i in range(len(picks))])   # first window of each row
        return {"ids": r["input_ids"].astype(np.int64),
                "labels": shifted_labels(r["input_ids"], r["doc_lens"], self.L),
                "pos": self.L.position_ids(r["doc_lens"], seq_len) if doc_mask else None,
                "group": group_ids(r), "names": [f"{s}/{t}" for s, t in zip(r["source"], r["row_type"])]}


# --------------------------------------------------------------------------------------
# prefetching (torch)
# --------------------------------------------------------------------------------------
class Prefetcher:
    """Background thread: reads + prepares whole steps ahead of time (pinned tensors).
    Recreate it after a rollback (it restarts from any data index in O(row group))."""

    def __init__(self, data: PackedData, start_index: int, depth: int, pin: bool):
        self.data, self.pin = data, pin
        self.q: queue.Queue = queue.Queue(maxsize=max(depth, 1))
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, args=(start_index,), daemon=True)
        self.thread.start()

    def _to_torch(self, a):
        import torch
        if a is None:
            return None
        t = torch.from_numpy(a)
        return t.pin_memory() if self.pin else t

    def _run(self, index: int) -> None:
        try:
            for di, batch in self.data.iter_steps(index):
                if self.stop.is_set():
                    return
                st = self.data.prepare(batch)
                st["index"] = di
                st["micro"] = [{k: self._to_torch(v) for k, v in m.items()} for m in st["micro"]]
                while not self.stop.is_set():
                    try:
                        self.q.put(st, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:  # surfaced on the next get()
            self.error = e

    def get(self, index: int) -> dict:
        while True:
            if self.error is not None:
                raise RuntimeError("data prefetch thread failed") from self.error
            try:
                st = self.q.get(timeout=1.0)
                break
            except queue.Empty:
                if not self.thread.is_alive() and self.error is None:
                    raise RuntimeError("data prefetch thread stopped")
        if st["index"] != index:
            raise RuntimeError(f"prefetch order mismatch: got data index {st['index']}, expected {index}")
        return st

    def close(self) -> None:
        self.stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.thread.join(timeout=10)
