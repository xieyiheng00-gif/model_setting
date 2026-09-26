"""API keys for this project, loaded into the process environment without ever being printed or logged.

Where the keys live (first match wins):
    $LMARCH_SECRETS_FILE                 explicit path (recommended on shared GPU nodes: a file in $HOME)
    <repo>/secrets.env                   gitignored; create it with  python scripts/setup_secrets.py
    ~/.config/lmarch/secrets.env
Format: KEY=value per line (see secrets.env.example). Variables already set in the environment win.

Keys used by the project (all optional; a missing key only disables what needs it):
    WANDB_API_KEY   W&B live logging (lmarch/train/wandb_sink.py reads it from the environment)
    HF_TOKEN        downloading the private packed dataset (`hf download xie1231/llm-10b-packed`)
    GITHUB_TOKEN    cloning / pushing the private repo from a GPU machine (git credential helper)

Safety: on POSIX the file must be private (mode 600; tightened automatically if it is group/world
readable), values are never placed on command lines, and redact() removes them from any text the run
logs (events, tracebacks, console echoes, wrapped command output).
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

KNOWN_KEYS = ("WANDB_API_KEY", "HF_TOKEN", "GITHUB_TOKEN")
REPO_ROOT = Path(__file__).resolve().parents[1]
_loaded_values: set[str] = set()


def candidate_files() -> list[Path]:
    out = []
    if os.environ.get("LMARCH_SECRETS_FILE"):
        out.append(Path(os.environ["LMARCH_SECRETS_FILE"]).expanduser())
    out += [REPO_ROOT / "secrets.env", Path.home() / ".config" / "lmarch" / "secrets.env"]
    return out


def secrets_file() -> Path | None:
    for p in candidate_files():
        if p.is_file():
            return p
    return None


def _ensure_private(path: Path) -> None:
    if os.name != "posix":
        return
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        try:
            os.chmod(path, 0o600)
            print(f"[secrets] {path} was readable by other users: permissions tightened to 600", file=sys.stderr)
        except OSError:
            raise PermissionError(f"{path} is readable by other users and cannot be chmod'ed to 600: fix it "
                                  "(chmod 600) before using it") from None


def parse(path: Path) -> dict[str, str]:
    """KEY=value lines; '#' comments; optional 'export ' prefix and surrounding quotes; no expansion."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if k and v:
            out[k] = v
    return out


def load_secrets(verbose: bool = False) -> dict[str, str]:
    """Put the file's keys into os.environ (existing variables win). Returns {KEY: 'file'|'env'|'missing'}
    for the known keys - names and origins only, never values."""
    path = secrets_file()
    values: dict[str, str] = {}
    if path is not None:
        _ensure_private(path)
        values = parse(path)
    status = {}
    for k, v in values.items():
        if k not in os.environ:
            os.environ[k] = v
            status[k] = "file"
    for k in KNOWN_KEYS:
        status.setdefault(k, "env" if os.environ.get(k) else "missing")
    for k in set(KNOWN_KEYS) | set(values):
        v = os.environ.get(k)
        if v and len(v) >= 6:
            _loaded_values.add(v)
    if verbose:
        where = str(path) if path else "no secrets file"
        print(f"[secrets] {where}: " + ", ".join(f"{k}={s}" for k, s in sorted(status.items())), file=sys.stderr)
    return status


def redact(text: str) -> str:
    """Replace any loaded secret value in `text` with ***."""
    if not _loaded_values or not isinstance(text, str):
        return text
    for v in _loaded_values:
        if v in text:
            text = text.replace(v, "***")
    return text
