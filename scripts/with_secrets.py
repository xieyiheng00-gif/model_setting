"""Run a command with the project's API keys in its environment. Everything it prints is masked.

  python scripts/with_secrets.py -- hf download xie1231/llm-10b-packed --repo-type dataset --local-dir /data/llm/packed
  python scripts/with_secrets.py -- wandb sync runs/<run>/wandb/offline-run-*
  python scripts/with_secrets.py -- git push            # git authenticates with GITHUB_TOKEN
  python scripts/with_secrets.py -- torchrun --standalone --nproc_per_node 8 scripts/train.py ...

The keys are loaded from the secrets file (see lmarch/secrets.py) into the CHILD's environment only -
never onto a command line (so they don't appear in `ps` or shell history). For git, a credential
helper reads $GITHUB_TOKEN from the environment. Every line the command prints goes through
redact(), so a key can't reach the terminal, a log file, or an assistant reading the output.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmarch.secrets import load_secrets, redact  # noqa: E402

GIT_HELPER = '!f() { test "$1" = get && echo username=x-access-token && echo "password=$GITHUB_TOKEN"; }; f'


def _pump(src, dst) -> None:
    for line in iter(src.readline, ""):
        dst.write(redact(line))
        dst.flush()
    src.close()


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    status = load_secrets()
    print("[with_secrets] " + ", ".join(f"{k}={v}" for k, v in sorted(status.items())), file=sys.stderr)
    env = dict(os.environ)
    if env.get("GITHUB_TOKEN"):
        # reset any configured helper (e.g. a desktop credential manager), then use the env-var helper
        n = int(env.get("GIT_CONFIG_COUNT", "0"))
        env.update({f"GIT_CONFIG_KEY_{n}": "credential.helper", f"GIT_CONFIG_VALUE_{n}": "",
                    f"GIT_CONFIG_KEY_{n + 1}": "credential.helper", f"GIT_CONFIG_VALUE_{n + 1}": GIT_HELPER,
                    "GIT_CONFIG_COUNT": str(n + 2), "GIT_TERMINAL_PROMPT": "0"})
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
    except FileNotFoundError:
        print(f"[with_secrets] command not found: {argv[0]}", file=sys.stderr)
        return 127
    threads = [threading.Thread(target=_pump, args=(p.stdout, sys.stdout), daemon=True),
               threading.Thread(target=_pump, args=(p.stderr, sys.stderr), daemon=True)]
    for t in threads:
        t.start()
    try:
        rc = p.wait()
    except KeyboardInterrupt:
        p.terminate()
        rc = p.wait()
    for t in threads:
        t.join(timeout=5)
    return rc


if __name__ == "__main__":
    sys.exit(main())
