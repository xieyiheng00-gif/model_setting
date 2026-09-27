"""Hugging Face checkpoint uploads: which checkpoints go up, the repo layout, retries, the marker files and the
re-queue after a restart. Offline (a fake Hub API); the retention test needs torch."""
import json
import types

import pytest

from lmarch.config import ConfigError, build_config
from lmarch.train import hf_upload as H


def _ckpt(run_dir, name, complete=True):
    p = run_dir / "checkpoints" / name
    p.mkdir(parents=True)
    (p / "model.pt").write_bytes(b"weights")
    (p / "optimizer.pt").write_bytes(b"adam")
    if complete:
        (p / "COMPLETE").write_text("1")
    return p


def _run_dir(tmp_path):
    rd = tmp_path / "run"
    (rd / "metrics").mkdir(parents=True)
    (rd / "meta.json").write_text("{}")
    (rd / "config.yaml").write_text("a: 1\n")
    (rd / "wandb_run_id.txt").write_text("abc123")
    (rd / "metrics" / "train_steps.jsonl").write_bytes(b'{"step": 0, "loss": 11.8}\n')
    return rd


class _Log:
    def __init__(self):
        self.events = []

    def event(self, kind, **d):
        self.events.append((kind, d))


def _uploader(rd, every=0, milestone=2500):
    ccfg = types.SimpleNamespace(hf_repo="me/ckpts", hf_private=True, hf_every=every, milestone_interval=milestone)
    return H.HFUploader(ccfg, rd, "kda_full_trunk_h100x8", _Log())


def test_which_checkpoints_are_uploaded(tmp_path):
    up = _uploader(_run_dir(tmp_path))
    assert up.wanted(2500, "") and up.wanted(12500, "") and up.wanted(13351, "final")
    assert not up.wanted(3000, "") and not up.wanted(2500, "emergency")
    up.logf.close()
    up = _uploader(_run_dir(tmp_path / "b"), every=500)
    assert up.wanted(3000, "") and not up.wanted(3250, "")
    up.logf.close()


def test_repo_layout_skips_markers_and_snapshots_run_files(tmp_path):
    rd = _run_dir(tmp_path)
    ck = _ckpt(rd, "step_0002500")
    (ck / H.PENDING).write_text("{}")
    ops = H.build_operations(ck, rd, "kda_full_trunk_h100x8")
    paths = sorted(o.path_in_repo for o in ops)
    assert paths == ["kda_full_trunk_h100x8/checkpoints/step_0002500/COMPLETE",
                     "kda_full_trunk_h100x8/checkpoints/step_0002500/model.pt",
                     "kda_full_trunk_h100x8/checkpoints/step_0002500/optimizer.pt",
                     "kda_full_trunk_h100x8/config.yaml",
                     "kda_full_trunk_h100x8/meta.json",
                     "kda_full_trunk_h100x8/metrics/train_steps.jsonl",
                     "kda_full_trunk_h100x8/wandb_run_id.txt"]
    metrics = next(o for o in ops if o.path_in_repo.endswith("train_steps.jsonl"))
    assert metrics.path_or_fileobj == b'{"step": 0, "loss": 11.8}\n'      # bytes: the live file may grow


def test_upload_creates_missing_repo_and_retries(tmp_path):
    from huggingface_hub.errors import RepositoryNotFoundError
    rd = _run_dir(tmp_path)
    ck = _ckpt(rd, "step_0005000")
    calls = {"create_repo": 0, "commit": 0}

    class Api:
        def repo_info(self, repo, repo_type):
            if calls["create_repo"] == 0:
                resp = types.SimpleNamespace(headers={}, status_code=404, request=None)
                raise RepositoryNotFoundError("missing", response=resp)

        def create_repo(self, repo, repo_type, private, exist_ok):
            assert private and exist_ok
            calls["create_repo"] += 1

        def create_commit(self, repo_id, repo_type, operations, commit_message):
            calls["commit"] += 1
            if calls["commit"] < 3:
                raise ConnectionError("flaky network")
            return types.SimpleNamespace(commit_url="https://huggingface.co/me/ckpts/commit/abc")

    url = H.upload_checkpoint("me/ckpts", ck, rd, "run", api=Api(), sleep=lambda s: None)
    assert url.endswith("/abc") and calls == {"create_repo": 1, "commit": 3}
    with pytest.raises(RuntimeError, match="flaky"):
        calls["commit"] = -10
        H.upload_checkpoint("me/ckpts", ck, rd, "run", api=Api(), attempts=2, sleep=lambda s: None)


def test_uploader_process_writes_markers(tmp_path, monkeypatch):
    rd = _run_dir(tmp_path)
    ck = _ckpt(rd, "step_0002500")
    (ck / H.PENDING).write_text("{}")
    monkeypatch.setattr(H, "upload_checkpoint", lambda *a, **k: "https://hf.co/commit/1")
    args = ["--repo", "me/ckpts", "--ckpt", str(ck), "--run_dir", str(rd), "--run_name", "r"]
    assert H.main(args) == 0
    assert json.loads((ck / H.DONE).read_text())["url"] == "https://hf.co/commit/1"
    assert not (ck / H.PENDING).exists()

    def boom(*a, **k):
        raise RuntimeError("403 Forbidden")
    monkeypatch.setattr(H, "upload_checkpoint", boom)
    assert H.main(args) == 1
    assert "403" in json.loads((ck / H.FAILED).read_text())["error"] and not (ck / H.PENDING).exists()
    assert [r["status"] for r in H.status(rd)] == ["failed"]


def test_restart_requeues_unfinished_uploads(tmp_path, monkeypatch):
    rd = _run_dir(tmp_path)
    for n in ("step_0002500", "step_0003000", "step_0013351_final"):
        _ckpt(rd, n)
    (_ckpt(rd, "step_0005000") / H.DONE).write_text("{}")
    _ckpt(rd, "step_0007500", complete=False)                    # interrupted save: ignored
    up = _uploader(rd)
    sent = []
    monkeypatch.setattr(up, "submit", lambda ck, step: sent.append((ck.name, step)))
    up.resume_pending(rd / "checkpoints")
    assert sent == [("step_0002500", 2500), ("step_0013351_final", 13351)]
    up.logf.close()


def test_config_validation():
    ok = build_config("configs/h100x8/trunk.yaml", arch="kda_full", overrides=["checkpoint.hf_repo=me/ckpts"])
    assert ok.checkpoint.hf_repo == "me/ckpts"
    for bad in (["checkpoint.hf_repo=no-slash"], ["checkpoint.hf_repo=me/ckpts", "checkpoint.hf_every=700"]):
        with pytest.raises(ConfigError):
            build_config("configs/h100x8/trunk.yaml", arch="kda_full", overrides=bad)


def test_rotation_keeps_checkpoints_that_are_still_uploading(tmp_path):
    pytest.importorskip("torch")
    from lmarch.train.checkpoint import CheckpointManager
    from lmarch.train.dist import DistInfo
    rd = tmp_path / "run"
    for s in (500, 1000, 1500, 2000, 3000):
        _ckpt(rd, f"step_{s:07d}")
    (rd / "checkpoints" / "step_0000500" / H.PENDING).write_text("{}")
    cm = CheckpointManager(rd / "checkpoints", DistInfo(), keep_last=2, milestone_interval=2500)
    cm._retention()
    left = sorted(p.name for p in (rd / "checkpoints").iterdir())
    assert left == ["step_0000500", "step_0002000", "step_0003000"]
