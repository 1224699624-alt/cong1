#!/usr/bin/env bash
# Corrected progress logging only; full matched run restarts from fresh100.
set -euo pipefail
cd /root/ram_h0_final_target_20260924
trap 'echo "$?" > FINAL_TARGET_EXIT_v2.txt' EXIT
test ! -e FIN_F0S1_v2
timeout 1100 /root/miniconda3/bin/python -u \
  code_final_v2/run_ram_h0_final_target.py \
  --lock FINAL_TARGET_RUN_LOCK_v2.json --arm SRC_F0S1 --out FIN_F0S1_v2 \
  > FIN_F0S1_v2.log 2>&1
