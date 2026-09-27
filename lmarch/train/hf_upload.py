"""Periodic checkpoint upload to the Hugging Face Hub, so a rented GPU machine can disappear without losing
the run.

What is uploaded (checkpoint.hf_repo set; rank 0 only):
    every checkpoint whose step is a multiple of checkpoint.hf_every (default: the milestone interval, i.e.
    the checkpoints that are never deleted locally) and the final one, each with a snapshot of the run's
    small files at that moment. Layout in the private repo:
        <run_name>/checkpoints/step_0002500/{model.pt, optimizer.pt, trainer_state.json, ...}
        <run_name>/{meta.json, config.yaml, wandb_run_id.txt, metrics/*.jsonl}
    The layout mirrors runs/<run_name>/, so  hf download <repo> --include "<run_name>/*" --local-dir runs
    restores a run on a new machine and the trainer resumes from its newest checkpoint.
How:
    every upload runs in its own process (python -m lmarch.train.hf_upload) in a new session, so it never
    slows the training loop and survives a supervisor restart of the trainer. Markers in the checkpoint
    directory record its state:
        .hf_pending    queued or running (local rotation keeps the checkpoint until the upload has finished)
        .hf_uploaded   done (JSON: repo, path, commit URL)
        .hf_failed     the last attempt failed (JSON: error); the next trainer start retries it
    At start-up the trainer re-queues checkpoints that should be on the Hub but are not marked done. At the
    normal end of a run it waits (up to checkpoint.hf_wait_min) for the uploads, so shutting the machine
    down right after training does not cut one off.
The token is HF_TOKEN from the environment (lmarch.secrets); it never appears on a command line or in a log.

Check a run by hand:   python -m lmarch.train.hf_upload --status runs/kda_full_trunk_h100x8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PENDING, DONE, FAILED = ".hf_pending", ".hf_uploaded", ".hf_failed"
LOG_NAME = "hf_upload.log"
RUN_FILES = ("meta.json", "config.yaml", "wandb_run_id.txt")   # + metrics/*.jsonl (the W&B id lets a restored run
                                                                 # continue the same W&B run)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0 or os.name == "nt":
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def checkpoint_step_tag(path: Path) -> tuple[int, str]:
    """step_0002500 -> (2500, ''); step_0013351_final -> (13351, 'final')."""
    parts = path.name.split("_", 2)
    return int(parts[1]), (parts[2] if len(parts) > 2 else "")


# --------------------------------------------------------------------------------------
# the upload itself (runs in the subprocess)
# --------------------------------------------------------------------------------------
def build_operations(ckpt: Path, run_dir: Path, run_name: str):
    """Commit operations: the checkpoint files (read from disk during the upload; they never change once
    COMPLETE is written) and a byte snapshot of the run's small files (these keep growing while training
    runs, so they are read once here)."""
    from huggingface_hub import CommitOperationAdd
    ops = []
    for f in sorted(ckpt.iterdir()):
        if f.is_file() and not f.name.startswith(".hf_"):
            ops.append(CommitOperationAdd(path_in_repo=f"{run_name}/checkpoints/{ckpt.name}/{f.name}",
                                          path_or_fileobj=str(f)))
    small = [run_dir / n for n in RUN_FILES] + sorted((run_dir / "metrics").glob("*.jsonl"))
    for f in small:
        if f.is_file():
            ops.append(CommitOperationAdd(path_in_repo=f"{run_name}/{f.relative_to(run_dir).as_posix()}",
                                          path_or_fileobj=f.read_bytes()))
    return ops


def ensure_repo(api, repo: str, private: bool) -> None:
    """Create the model repo if it does not exist (private by default). An existing repo is used as is, so a
    token that may write to the repo but not create repos still works."""
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:                              # huggingface_hub < 0.23
        from huggingface_hub.utils import RepositoryNotFoundError
    try:
        api.repo_info(repo, repo_type="model")
    except RepositoryNotFoundError:
        api.create_repo(repo, repo_type="model", private=private, exist_ok=True)


def upload_checkpoint(repo: str, ckpt: Path, run_dir: Path, run_name: str, private: bool = True,
                      attempts: int = 4, api=None, sleep=time.sleep) -> str:
    """One commit with the checkpoint + run files; retried with backoff. Returns the commit URL."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    last = None
    for i in range(attempts):
        try:
            ensure_repo(api, repo, private)
            ops = build_operations(ckpt, run_dir, run_name)
            info = api.create_commit(repo_id=repo, repo_type="model", operations=ops,
                                     commit_message=f"{run_name}: {ckpt.name}")
            return str(getattr(info, "commit_url", "") or info)
        except Exception as e:  # noqa: BLE001 - network / server errors: back off and retry
            last = e
            if i + 1 < attempts:
                wait = 30 * 2 ** i
                print(f"[hf_upload] {ckpt.name}: attempt {i + 1}/{attempts} failed ({type(e).__name__}: {e}); "
                      f"retrying in {wait}s", flush=True)
                sleep(wait)
    raise RuntimeError(f"{type(last).__name__}: {last}") from last


def _upload_main(a) -> int:
    from ..secrets import load_secrets, redact
    load_secrets()                                   # HF_TOKEN is inherited; this only arms redact()
    ckpt, run_dir = Path(a.ckpt), Path(a.run_dir)
    t0 = time.time()
    for m in (DONE, FAILED):                         # markers of an earlier upload of this directory are stale
        (ckpt / m).unlink(missing_ok=True)
    _write_json(ckpt / PENDING, {"repo": a.repo, "pid": os.getpid(), "started": t0})
    print(f"[hf_upload] {time.strftime('%H:%M:%S')} start {ckpt.name} -> {a.repo}/{a.run_name}", flush=True)
    try:
        url = upload_checkpoint(a.repo, ckpt, run_dir, a.run_name, private=not a.public)
    except Exception as e:  # noqa: BLE001
        msg = redact(f"{type(e).__name__}: {e}")[:2000]
        _write_json(ckpt / FAILED, {"repo": a.repo, "error": msg, "time": time.time()})
        (ckpt / PENDING).unlink(missing_ok=True)
        print(f"[hf_upload] FAILED {ckpt.name}: {msg}", flush=True)
        return 1
    _write_json(ckpt / DONE, {"repo": a.repo, "path": f"{a.run_name}/checkpoints/{ckpt.name}", "url": url,
                              "seconds": round(time.time() - t0, 1), "time": time.time()})
    (ckpt / FAILED).unlink(missing_ok=True)
    (ckpt / PENDING).unlink(missing_ok=True)
    print(f"[hf_upload] {time.strftime('%H:%M:%S')} done {ckpt.name} in {time.time() - t0:.0f}s: {url}", flush=True)
    return 0


def status(run_dir: Path) -> list[dict]:
    out = []
    for p in sorted((run_dir / "checkpoints").glob("step_*")):
        if not p.is_dir() or p.name.endswith(".tmp"):
            continue
        st = "uploaded" if (p / DONE).exists() else "pending" if (p / PENDING).exists() else \
            "failed" if (p / FAILED).exists() else "local only"
        out.append({"checkpoint": p.name, "status": st, **_read_json(p / DONE), **_read_json(p / FAILED)})
    return out


# --------------------------------------------------------------------------------------
# the trainer side (rank 0)
# --------------------------------------------------------------------------------------
class HFUploader:
    def __init__(self, ccfg, run_dir: Path, run_name: str, log):
        self.repo, self.private = ccfg.hf_repo, ccfg.hf_private
        self.every = ccfg.hf_every or ccfg.milestone_interval
        self.run_dir, self.run_name, self.log = Path(run_dir), run_name, log
        self.jobs: dict[str, tuple] = {}             # ckpt path -> (Popen, start time, step, ckpt)
        self.logf = open(self.run_dir / LOG_NAME, "a", encoding="utf-8")

    def wanted(self, step: int, tag: str) -> bool:
        return tag == "final" or (tag == "" and self.every > 0 and step % self.every == 0)

    def submit(self, ckpt: Path, step: int) -> None:
        ckpt = Path(ckpt)
        job = self.jobs.get(str(ckpt))
        if job is not None and job[0].poll() is None:
            return                                   # already uploading
        for m in (DONE, FAILED):
            (ckpt / m).unlink(missing_ok=True)
        _write_json(ckpt / PENDING, {"repo": self.repo, "queued": time.time()})   # the uploader adds its pid
        cmd = [sys.executable, "-m", "lmarch.train.hf_upload", "--repo", self.repo, "--ckpt", str(ckpt),
               "--run_dir", str(self.run_dir), "--run_name", self.run_name] + ([] if self.private else ["--public"])
        kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        p = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), **kw)
        self.jobs[str(ckpt)] = (p, time.time(), step, ckpt)
        self.log.event("hf_upload", status="started", step=step, checkpoint=ckpt.name, repo=self.repo)

    def on_save(self, ckpt: Path, step: int, tag: str) -> None:
        if self.wanted(step, tag):
            self.submit(ckpt, step)

    def resume_pending(self, ckpt_dir: Path) -> None:
        """Re-queue checkpoints that should be on the Hub but are not marked uploaded (crash, restart)."""
        for p in sorted(Path(ckpt_dir).glob("step_*")):
            if not p.is_dir() or p.name.endswith(".tmp") or not (p / "COMPLETE").exists() or (p / DONE).exists():
                continue
            try:
                step, tag = checkpoint_step_tag(p)
            except (IndexError, ValueError):
                continue
            if not self.wanted(step, tag):
                continue
            if _alive(_read_json(p / PENDING).get("pid")):
                continue                             # an uploader from the previous trainer is still running
            self.submit(p, step)

    def poll(self) -> None:
        for key, (p, t0, step, ckpt) in list(self.jobs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del self.jobs[key]
            if rc == 0 and (ckpt / DONE).exists():
                d = _read_json(ckpt / DONE)
                self.log.event("hf_upload", status="done", step=step, checkpoint=ckpt.name,
                               seconds=round(time.time() - t0, 1), url=d.get("url"))
            else:
                err = _read_json(ckpt / FAILED).get("error") or f"uploader exit code {rc} (see {LOG_NAME})"
                if (ckpt / PENDING).exists():          # the uploader died before cleaning up
                    (ckpt / PENDING).unlink(missing_ok=True)
                    _write_json(ckpt / FAILED, {"repo": self.repo, "error": err, "time": time.time()})
                self.log.event("hf_upload_failed", step=step, checkpoint=ckpt.name, error=err)

    def finish(self, wait_s: float) -> None:
        """Wait for running uploads (normal end of a run); anything still running keeps going on its own."""
        self.poll()
        if self.jobs and wait_s > 0:
            print(f"[hf_upload] waiting up to {wait_s / 60:.0f} min for {len(self.jobs)} upload(s) to "
                  f"{self.repo} ...", flush=True)
            deadline = time.time() + wait_s
            while self.jobs and time.time() < deadline:
                time.sleep(5)
                self.poll()
        for p, _, step, ckpt in self.jobs.values():
            self.log.event("hf_upload", status="still_running", step=step, checkpoint=ckpt.name, pid=p.pid,
                           message=f"continues in the background; check with python -m lmarch.train.hf_upload "
                                   f"--status {self.run_dir}")
        self.logf.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Upload one checkpoint to the Hugging Face Hub, or show upload status.")
    ap.add_argument("--status", metavar="RUN_DIR", help="print the upload state of every checkpoint of a run")
    ap.add_argument("--repo")
    ap.add_argument("--ckpt")
    ap.add_argument("--run_dir")
    ap.add_argument("--run_name")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args(argv)
    if a.status:
        for r in status(Path(a.status)):
            print(f"{r['checkpoint']:24s} {r['status']:10s} {r.get('url') or r.get('error') or ''}")
        return 0
    if not (a.repo and a.ckpt and a.run_dir and a.run_name):
        ap.error("--repo, --ckpt, --run_dir and --run_name are required")
    return _upload_main(a)


if __name__ == "__main__":
    sys.exit(main())
