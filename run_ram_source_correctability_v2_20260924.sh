#!/usr/bin/env bash
# Complete four-arm restart after the first launcher stopped before update 1.
set -euo pipefail
cd /root/ram_source_correctability_20260924
trap 'echo "$?" > FOUR_ARM_EXIT_v2.txt' EXIT
test ! -e v2
mkdir v2
for arm in SRC_F0S0 SRC_F0S1 SRC_F1S0 SRC_F1S1; do
  timeout 900 /root/miniconda3/bin/python -u code_training_v4/run_ram_source_correctability_2x2.py \
    --lock SOURCE_CORRECTABILITY_RUN_LOCK_v2.json --arm "$arm" --out "v2/$arm" \
    > "v2/$arm.log" 2>&1
done
