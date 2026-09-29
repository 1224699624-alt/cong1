#!/usr/bin/env bash
# Four arms from one C0, with the validated schedule-step metadata contract.
set -euo pipefail
cd /root/ram_source_correctability_20260924
trap 'echo "$?" > FOUR_ARM_EXIT_v3.txt' EXIT
test ! -e v3
mkdir v3
for arm in SRC_F0S0 SRC_F0S1 SRC_F1S0 SRC_F1S1; do
  timeout 900 /root/miniconda3/bin/python -u code_training_v5/run_ram_source_correctability_2x2.py \
    --lock SOURCE_CORRECTABILITY_RUN_LOCK_v3.json --arm "$arm" --out "v3/$arm" \
    > "v3/$arm.log" 2>&1
done
