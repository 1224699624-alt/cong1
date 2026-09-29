#!/usr/bin/env bash
# Four locked arms. Each run writes to its own absent directory.
set -euo pipefail
cd /root/ram_source_correctability_20260924
trap 'echo "$?" > FOUR_ARM_EXIT.txt' EXIT
for arm in SRC_F0S0 SRC_F0S1 SRC_F1S0 SRC_F1S1; do
  test ! -e "$arm"
done
for arm in SRC_F0S0 SRC_F0S1 SRC_F1S0 SRC_F1S1; do
  timeout 900 /root/miniconda3/bin/python -u code_training_v3/run_ram_source_correctability_2x2.py \
    --lock SOURCE_CORRECTABILITY_RUN_LOCK.json --arm "$arm" --out "$arm" \
    > "$arm.log" 2>&1
done
