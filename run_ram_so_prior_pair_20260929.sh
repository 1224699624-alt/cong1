#!/usr/bin/env bash
set -euo pipefail
cd /root/ram_so_heatmap_20260929
export PYTHONPATH=/root/ram_so_heatmap_20260929/code_v1
export OMP_NUM_THREADS=8
PY=/root/miniconda3/bin/python
LOCK=SO_PRIOR_RUN_LOCK.json
START=$(date +%s)
for ARM in O_REF_CONTINUE O_HARD_WEIGHT; do
  "$PY" code_v1/run_ram_so_o_pair.py --phase train --arm "$ARM" --lock "$LOCK" > "${ARM}.log" 2>&1
done
"$PY" code_v1/run_ram_so_o_pair.py --phase eval --lock "$LOCK" > EVALUATION.log 2>&1
END=$(date +%s)
echo "$((END-START))" > PAIR_WALL_SECONDS.txt
echo COMPLETE > PAIR_STATUS.txt
