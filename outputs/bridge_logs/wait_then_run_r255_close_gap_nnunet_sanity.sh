#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
WAIT_LOG="$PWD/outputs/bridge_logs/r255_close_gap_gpu_wait.log"
echo "[R255-WAIT] started $(date -Is)" >> "$WAIT_LOG"
while true; do
  USED="$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  UTIL="$(nvidia-smi --id=1 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
  echo "[R255-WAIT] $(date -Is) gpu1 memory=${USED}MiB util=${UTIL}%" >> "$WAIT_LOG"
  if [ "$USED" -lt 500 ]; then
    break
  fi
  sleep 30
done
echo "[R255-WAIT] GPU1 free; launching sanity $(date -Is)" >> "$WAIT_LOG"
exec bash outputs/bridge_logs/run_r255_close_gap_nnunet_sanity.sh
