#!/usr/bin/env bash
# The full plan for all five architectures, each run under the crash supervisor:
#   stage 1 (trunk, 13,351 steps) -> branch point -> stage 2 16K branch + 4K branch (5,419 steps each).
#   bash scripts/run_all.sh h100x8                 # torchrun, 8 GPUs
#   bash scripts/run_all.sh h100x1
#   ARCHS="dense kda_dsa" STAGES="trunk s2_16k" bash scripts/run_all.sh h100x8
set -uo pipefail
HW=${1:-h100x8}
ARCHS=${ARCHS:-"dense dsa kda_full kda_dsa csa"}
STAGES=${STAGES:-"trunk s2_16k s2_4k"}
cd "$(dirname "$0")/.."
if [ "$HW" = "h100x8" ]; then
  LAUNCH=(torchrun --standalone --nproc_per_node 8 scripts/train.py)
else
  LAUNCH=(python scripts/train.py)
fi
for ARCH in $ARCHS; do
  for STAGE in $STAGES; do
    RUN="${ARCH}_${STAGE}_${HW}"
    python scripts/supervise.py --run_dir "runs/${RUN}" -- \
      "${LAUNCH[@]}" --config "configs/${HW}/${STAGE}.yaml" --arch "$ARCH"
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "!! ${RUN} ended with code ${rc} - see runs/${RUN}/metrics/events.jsonl"
      [ "$STAGE" = "trunk" ] && break      # no branch point: skip this arch's stage-2 runs
    fi
  done
done
python scripts/analyze_runs.py runs/*_trunk_"${HW}" --out "reports/${HW}_trunk"
python scripts/analyze_runs.py runs/*_s2_16k_"${HW}" runs/*_s2_4k_"${HW}" --out "reports/${HW}_stage2"
