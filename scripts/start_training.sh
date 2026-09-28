#!/usr/bin/env bash
# Start training inside a detached tmux session: it keeps running when SSH drops or you close your laptop.
#
#   bash scripts/start_training.sh                          # kda_full: trunk -> 16K + 4K branches, 8 GPUs
#   ARCHS="kda_full dense" bash scripts/start_training.sh h100x8
#   STAGES="s2_16k s2_4k" bash scripts/start_training.sh    # only the branches (the trunk run already finished)
#
# The tmux session (default name "train") gets two windows:
#   train   scripts/run_all.sh (each stage under the crash supervisor); everything it prints also goes to
#           logs/train_<date>.log
#   gpu     nvitop (or nvidia-smi): utilization, memory, power per GPU
# Useful keys:  tmux attach -t train   look at it      Ctrl-b d          detach (the run keeps going)
#               Ctrl-b n / Ctrl-b p    next / previous window
set -euo pipefail
HW=${1:-h100x8}
ARCHS=${ARCHS:-kda_full}
STAGES=${STAGES:-"trunk s2_16k s2_4k"}
SESSION=${SESSION:-train}
cd "$(dirname "$0")/.."
ROOT=$PWD

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed: apt-get update && apt-get install -y tmux" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "a tmux session '$SESSION' already exists: tmux attach -t $SESSION   (or SESSION=train2 bash $0)" >&2
  exit 1
fi
if [ -z "${LMARCH_SECRETS_FILE:-}" ] && [ -f /dev/shm/lmarch.env ]; then
  export LMARCH_SECRETS_FILE=/dev/shm/lmarch.env        # the keys sent into RAM (README checklist, step B1)
fi

# the keys training needs: W&B (live curves) and Hugging Face (data, checkpoint uploads). Names only are printed.
python - <<'PY'
import sys
sys.path.insert(0, ".")
from lmarch.secrets import load_secrets
missing = [k for k, v in load_secrets().items() if v == "missing" and k in ("WANDB_API_KEY", "HF_TOKEN")]
if missing:
    sys.exit("missing keys: " + ", ".join(missing) + " - send them first (README checklist, step B1)")
PY
python scripts/setup_secrets.py --verify || { echo "a key was rejected (see above): fix it before training" >&2; exit 1; }

mkdir -p logs
LOG="logs/train_$(date +%Y%m%d_%H%M%S).log"
RUN="cd '$ROOT' && export PYTHONUNBUFFERED=1 LMARCH_SECRETS_FILE='${LMARCH_SECRETS_FILE:-}' ARCHS='$ARCHS' STAGES='$STAGES' && bash scripts/run_all.sh $HW 2>&1 | tee -a '$LOG'"
tmux new-session -d -s "$SESSION" -n train -c "$ROOT"
tmux set-option -t "$SESSION" history-limit 200000 >/dev/null
tmux send-keys -t "$SESSION:train" "$RUN" C-m
if command -v nvitop >/dev/null 2>&1; then GPU_VIEW="nvitop"; else GPU_VIEW="watch -n 1 nvidia-smi"; fi
tmux new-window -d -t "$SESSION" -n gpu -c "$ROOT" "$GPU_VIEW"

echo "training started in tmux session '$SESSION': $ARCHS | $STAGES | $HW"
echo "  log file:  $LOG      (the W&B link: grep -m3 'wandb.ai' $LOG)"
echo "  look:      tmux attach -t $SESSION     detach: Ctrl-b d     GPU window: Ctrl-b n"
echo "You can close SSH and this computer now; the run keeps going on the server."
