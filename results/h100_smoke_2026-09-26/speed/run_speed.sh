#!/usr/bin/env bash
# Full-size speed test on 1 x H100: for each arch, 30 trunk steps (4K) then 20 stage-2 steps (16K, branched from
# that trunk), with a 1 Hz nvidia-smi trace for power/clock/utilization. Usage: bash run_speed.sh [arch ...]
set -u
HERE=/workspace/h100_smoke
cd /workspace/model_setting
set -a; . ./secrets.env; set +a
ARCHS=${*:-dense dsa kda_full kda_dsa csa}
nvidia-smi --query-gpu=timestamp,clocks.sm,utilization.gpu,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -lms 1000 >> $HERE/gpu_trace.csv &
TRACE=$!
for arch in $ARCHS; do
  for stage in trunk s2_16k; do
    echo "=== $arch / $stage start $(date +%T)"
    python scripts/train.py --config $HERE/configs/$stage.yaml --arch $arch \
      --set log.out_dir=$HERE/runs_speed log.wandb_mode=offline "log.wandb_tags=[speedtest]" \
      > $HERE/logs/speed_${arch}_${stage}.log 2>&1
    rc=$?
    echo "=== $arch / $stage exit $rc $(date +%T) | $(grep -E '^step' $HERE/logs/speed_${arch}_${stage}.log | tail -1)"
    [ $rc -ne 0 ] && tail -5 $HERE/logs/speed_${arch}_${stage}.log && [ $stage = trunk ] && break
  done
done
kill $TRACE
echo "=== SPEED DONE $(date +%T)"
