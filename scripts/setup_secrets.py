"""Create or update the project's secrets file with hidden input (values are never echoed or printed).

  python scripts/setup_secrets.py              # prompt for each key (input hidden; Enter keeps the current value)
  python scripts/setup_secrets.py --check      # which keys are available (names only)
  python scripts/setup_secrets.py --verify     # ask each service who the key belongs to (account names only)
  python scripts/setup_secrets.py --path ~/.config/lmarch/secrets.env   # keep the file outside the repo

Run it on every machine that trains (the RTX 3060 box, the H100 node). On a shared GPU node prefer a
file in your home directory and `export LMARCH_SECRETS_FILE=~/.config/lmarch/secrets.env`.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmarch.secrets import KNOWN_KEYS, REPO_ROOT, load_secrets, parse, redact  # noqa: E402

HELP = {
    "WANDB_API_KEY": "W&B key (https://wandb.ai/authorize)",
    "HF_TOKEN": "Hugging Face read token (https://huggingface.co/settings/tokens)",
    "GITHUB_TOKEN": "GitHub fine-grained token for xieyiheng00-gif/model_setting",
}
DATASET = "xie1231/llm-10b-packed"
REPO = "xieyiheng00-gif/model_setting"


def target_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    if os.environ.get("LMARCH_SECRETS_FILE"):
        return Path(os.environ["LMARCH_SECRETS_FILE"]).expanduser()
    return REPO_ROOT / "secrets.env"


def write_private(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# written by scripts/setup_secrets.py - private, never commit or share this file"]
    lines += [f"{k}={values.get(k, '')}" for k in KNOWN_KEYS]
    lines += [f"{k}={v}" for k, v in values.items() if k not in KNOWN_KEYS]
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    if os.name == "posix":
        os.chmod(path, 0o600)


def check_gitignored(path: Path) -> None:
    try:
        path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return                                  # outside the repo: nothing to check
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", str(path)], capture_output=True)
    if r.returncode != 0:
        print(f"WARNING: {path} is NOT gitignored - add it to .gitignore before committing anything", file=sys.stderr)


def cmd_setup(path: Path) -> None:
    current = parse(path) if path.is_file() else {}
    print(f"secrets file: {path}\nInput is hidden. Press Enter to keep the current value (or leave it empty).")
    new = dict(current)
    for k in KNOWN_KEYS:
        state = "set" if current.get(k) else "empty"
        v = getpass.getpass(f"  {k} [{state}] - {HELP[k]}: ").strip()
        if v:
            new[k] = v
    write_private(path, new)
    check_gitignored(path)
    print("saved (" + ", ".join(f"{k}={'set' if new.get(k) else 'empty'}" for k in KNOWN_KEYS) + ")")


def _get_json(url: str, token: str, scheme: str = "Bearer") -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"{scheme} {token}", "User-Agent": "lmarch"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def cmd_verify() -> int:
    ok = True
    tok = os.environ.get("WANDB_API_KEY")
    if tok:
        try:
            import wandb
            v = wandb.Api(api_key=tok).viewer
            print(f"WANDB_API_KEY  ok: user '{getattr(v, 'username', '?')}', entity '{getattr(v, 'entity', '?')}'")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"WANDB_API_KEY  FAILED: {redact(f'{type(e).__name__}: {e}')[:200]}")
    tok = os.environ.get("HF_TOKEN")
    if tok:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=tok)
            who = api.whoami().get("name", "?")
            api.dataset_info(DATASET)
            print(f"HF_TOKEN       ok: user '{who}', can read dataset {DATASET}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"HF_TOKEN       FAILED: {redact(f'{type(e).__name__}: {e}')[:200]}")
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        try:
            login = _get_json("https://api.github.com/user", tok).get("login", "?")
            perms = _get_json(f"https://api.github.com/repos/{REPO}", tok).get("permissions", {})
            print(f"GITHUB_TOKEN   ok: user '{login}', {REPO}: pull={perms.get('pull')} push={perms.get('push')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"GITHUB_TOKEN   FAILED: {redact(f'{type(e).__name__}: {e}')[:200]}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.path:
        os.environ["LMARCH_SECRETS_FILE"] = str(Path(a.path).expanduser())
    if not (a.check or a.verify):
        cmd_setup(target_path(a.path))
    status = load_secrets()
    print("keys: " + ", ".join(f"{k}={status[k]}" for k in KNOWN_KEYS) + "   (file = secrets file, env = already exported)")
    return cmd_verify() if a.verify else 0


if __name__ == "__main__":
    sys.exit(main())
