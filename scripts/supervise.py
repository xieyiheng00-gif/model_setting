"""Crash supervisor: run a training command, restart it with auto-resume when it fails.

  python scripts/supervise.py --run_dir runs/dsa_h100x8 -- \
      torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/h100x8.yaml --arch dsa
  python scripts/supervise.py --run_dir runs/kda_dsa_rtx3060 -- \
      python scripts/train.py --config configs/rtx3060.yaml --arch kda_dsa

Policy (exit code of the trainer, read from <run_dir>/exit_status_rank*.json because torchrun hides
the worker's own code):
  0 finished                  -> stop
  2 config / data error       -> stop (fix the config)
  4 diverged (rollbacks used) -> stop (needs a human: lower LR, enable qk_norm, inspect crash_reports/)
  6 user stop                 -> stop
  3 CUDA OOM                  -> restart with train.micro_batch_size halved (grad-accum doubles, same global batch)
  5 preempted (SIGTERM)       -> restart immediately
  1 / killed / hung           -> restart with exponential backoff
Hang detection: if <run_dir>/heartbeat.json is older than --heartbeat_timeout the process tree is killed
(NCCL deadlock, dead node, stuck filesystem) and restarted. A crash loop (3 failures without progress)
stops the supervisor.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lmarch.secrets import load_secrets  # noqa: E402

SEVERITY = {3: 6, 4: 5, 2: 4, 1: 3, 5: 2, 6: 1, 0: 0}   # which rank status wins


class Supervisor:
    def __init__(self, a):
        self.a = a
        self.run_dir = Path(a.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "supervisor_events.jsonl"
        self.proc: subprocess.Popen | None = None
        self.stopping = False

    def log(self, kind: str, **kw) -> None:
        rec = {"time": time.time(), "kind": kind, **kw}
        print(f"[supervisor] {kind} {kw}", flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _status(self, returncode: int) -> dict:
        best = None
        for f in self.run_dir.glob("exit_status_rank*.json"):
            try:
                with open(f, encoding="utf-8") as fh:
                    st = json.load(fh)
            except Exception:
                continue
            if best is None or SEVERITY.get(st.get("code"), 3) > SEVERITY.get(best.get("code"), 3):
                best = st
        if best is None:
            best = {"code": returncode if returncode in SEVERITY else 1, "reason": f"no status file (rc={returncode})"}
        return best

    def _heartbeat_age(self) -> tuple[float | None, int | None]:
        p = self.run_dir / "heartbeat.json"
        try:
            with open(p, encoding="utf-8") as f:
                hb = json.load(f)
            return time.time() - float(hb["time"]), int(hb["step"])
        except Exception:
            return None, None

    def _kill_tree(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                for _ in range(120):
                    if self.proc.poll() is not None:
                        return
                    time.sleep(0.5)
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _on_signal(self, signum, frame):
        self.stopping = True
        self.log("supervisor_signal", signum=signum)
        if self.proc is not None and self.proc.poll() is None:
            try:  # forward: the trainer checkpoints and exits cleanly
                if os.name == "nt":
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                pass

    def run(self) -> int:
        a = self.a
        cmd = list(a.cmd)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            print("no command given (put it after --)", file=sys.stderr)
            return 2
        if "--resume" not in cmd:
            cmd += ["--resume", "auto"]
        load_secrets(verbose=True)          # restarted children inherit the keys (env only, never argv)
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        micro = None
        restarts, fails_no_progress, last_fail_step = 0, 0, None
        while True:
            for f in list(self.run_dir.glob("exit_status_rank*.json")) + [self.run_dir / "heartbeat.json"]:
                f.unlink(missing_ok=True)
            full = cmd + (["--set", f"train.micro_batch_size={micro}"] if micro else [])
            self.log("launch", attempt=restarts, cmd=" ".join(full))
            kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            t_start = time.time()
            self.proc = subprocess.Popen(full, **kw)
            hung = False
            while self.proc.poll() is None:
                time.sleep(10)
                age, _ = self._heartbeat_age()
                started = time.time() - t_start
                if age is not None and age > a.heartbeat_timeout and started > a.heartbeat_timeout:
                    self.log("hang_detected", heartbeat_age_s=round(age))
                    hung = True
                    self._kill_tree()
                    break
                if age is None and started > a.startup_timeout:
                    self.log("startup_timeout", seconds=round(started))
                    hung = True
                    self._kill_tree()
                    break
            rc = self.proc.wait()
            st = {"code": 1, "reason": "hang"} if hung else self._status(rc)
            code = int(st.get("code", 1))
            self.log("exited", returncode=rc, code=code, reason=st.get("reason"), step=st.get("step"))
            if self.stopping or code in (0, 2, 4, 6):
                return code
            restarts += 1
            if restarts > a.max_restarts:
                self.log("give_up", reason="max_restarts reached")
                return code
            step = st.get("step") if st.get("step") is not None else self._heartbeat_age()[1]
            if step is not None and step == last_fail_step:
                fails_no_progress += 1
            else:
                fails_no_progress = 0
            last_fail_step = step
            if fails_no_progress >= a.max_fails_without_progress - 1 and code != 3:
                self.log("give_up", reason=f"crash loop: {a.max_fails_without_progress} failures at step {step}")
                return code
            if code == 3:
                cur = int(st.get("micro_batch_size") or micro or 0)
                if cur <= 1:
                    self.log("give_up", reason="OOM at micro_batch_size=1")
                    return code
                micro = max(1, cur // 2)
                self.log("oom_retry", new_micro_batch_size=micro)
                delay = 5
            elif code == 5:
                delay = 5
            else:
                delay = min(a.backoff_base * 2 ** min(fails_no_progress, 5), a.backoff_max)
            self.log("restart_in", seconds=delay)
            time.sleep(delay)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="the run directory the trainer writes (runs/<run_name>)")
    ap.add_argument("--max_restarts", type=int, default=20)
    ap.add_argument("--heartbeat_timeout", type=float, default=1800.0)
    ap.add_argument("--startup_timeout", type=float, default=3600.0)
    ap.add_argument("--max_fails_without_progress", type=int, default=3)
    ap.add_argument("--backoff_base", type=float, default=30.0)
    ap.add_argument("--backoff_max", type=float, default=600.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    return Supervisor(ap.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
