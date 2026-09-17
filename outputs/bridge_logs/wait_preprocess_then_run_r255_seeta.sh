#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src
LOG="$PWD/outputs/bridge_logs/r255_seeta_chain.log"
echo "[R255-SEETA] chain started $(date -Is)" >> "$LOG"
while screen -list 2>/dev/null | grep -q '[.]r255_preprocess'; do
  echo "[R255-SEETA] waiting for preprocessing $(date -Is)" >> "$LOG"
  sleep 30
done
if ! grep -q '\[R255-SEETA-PREPROCESS\] done' outputs/bridge_logs/r255_seeta_preprocess.log; then
  echo "[R255-SEETA] preprocessing failed; training not launched $(date -Is)" >> "$LOG"
  exit 2
fi
CHECKPOINT="outputs/nnunet/r202/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
printf '%s  %s\n' '65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7' "$CHECKPOINT" | sha256sum -c -
USED="$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [ "$USED" -ge 500 ]; then
  echo "[R255-SEETA] GPU0 unexpectedly busy (${USED}MiB); refusing launch" >> "$LOG"
  exit 3
fi
echo "[R255-SEETA] preprocessing and checkpoint verified; launching $(date -Is)" >> "$LOG"
exec bash outputs/bridge_logs/run_r255_close_gap_nnunet_sanity_seeta.sh
