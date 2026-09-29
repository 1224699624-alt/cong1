#!/usr/bin/env bash
# One pre-locked RAM F0S1 target-location variant; no old queue continuation.
set -euo pipefail
cd /root/ram_h0_final_target_20260924
trap 'echo "$?" > FINAL_TARGET_EXIT_v1.txt' EXIT
test ! -e FIN_F0S1
timeout 1100 /root/miniconda3/bin/python -u \
  code_final_v1/run_ram_h0_final_target.py \
  --lock FINAL_TARGET_RUN_LOCK_v1.json --arm SRC_F0S1 --out FIN_F0S1 \
  > FIN_F0S1.log 2>&1
