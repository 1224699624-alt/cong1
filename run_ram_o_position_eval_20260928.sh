#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ( "$1" != val && "$1" != test ) ]]; then
  echo 'usage: run_ram_o_position_eval_20260928.sh val|test' >&2
  exit 2
fi
SPLIT="$1"
ROOT=/root/ram_o_position_20260928
PYTHON=/root/miniconda3/bin/python
export PYTHONPATH="$ROOT/code_v1"
export OMP_NUM_THREADS=16
cd "$ROOT"
test "$(cat TRAIN_STATUS.txt)" = ALL_TRAIN_ARMS_COMPLETE
if [[ "$SPLIT" == test ]]; then
  test "$(cat VAL_STATUS.txt)" = ALL_VAL_ARMS_COMPLETE
  test -f TEST_EVAL_LOCK.json
fi
for ARM in C0 ZERO INPUT_O REFINER_O; do
  echo "START $SPLIT $ARM $(date -u +%FT%TZ)" | tee -a EVAL_SEQUENCE.log
  if [[ "$ARM" == C0 ]]; then
    "$PYTHON" code_v1/run_ram_o_position_pair.py \
      --phase eval --arm "$ARM" --split "$SPLIT" \
      --lock POSITION_RUN_LOCK_v1.json \
      --out "results/EVAL_${SPLIT^^}_$ARM" \
      > "results/EVAL_${SPLIT^^}_$ARM.log" 2>&1
  else
    "$PYTHON" code_v1/run_ram_o_position_pair.py \
      --phase eval --arm "$ARM" --split "$SPLIT" \
      --checkpoint "results/TRAIN_$ARM/final.pth" \
      --lock POSITION_RUN_LOCK_v1.json \
      --out "results/EVAL_${SPLIT^^}_$ARM" \
      > "results/EVAL_${SPLIT^^}_$ARM.log" 2>&1
  fi
  echo "COMPLETE $SPLIT $ARM $(date -u +%FT%TZ)" | tee -a EVAL_SEQUENCE.log
done
if [[ "$SPLIT" == val ]]; then
  echo ALL_VAL_ARMS_COMPLETE > VAL_STATUS.txt
else
  echo ALL_TEST_ARMS_COMPLETE > TEST_STATUS.txt
fi
