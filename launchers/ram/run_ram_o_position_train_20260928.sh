#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/ram_o_position_20260928
PYTHON=/root/miniconda3/bin/python
export PYTHONPATH="$ROOT/code_v1"
export OMP_NUM_THREADS=16
cd "$ROOT"
for ARM in ZERO INPUT_O REFINER_O; do
  echo "START $ARM $(date -u +%FT%TZ)" | tee -a TRAIN_SEQUENCE.log
  "$PYTHON" code_v1/run_ram_o_position_pair.py \
    --phase train --arm "$ARM" --lock POSITION_RUN_LOCK_v1.json \
    --out "results/TRAIN_$ARM" > "results/TRAIN_$ARM.log" 2>&1
  echo "COMPLETE $ARM $(date -u +%FT%TZ)" | tee -a TRAIN_SEQUENCE.log
done
echo 'ALL_TRAIN_ARMS_COMPLETE' > TRAIN_STATUS.txt
