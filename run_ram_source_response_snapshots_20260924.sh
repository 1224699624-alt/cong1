#!/usr/bin/env bash
# Read-only pre-registered step-1 and step-192 responses; no optimizer steps.
set -euo pipefail
cd /root/ram_source_correctability_20260924
trap 'echo "$?" > RESPONSE_SNAPSHOTS_EXIT.txt' EXIT
test -f response_v4/RESULT.json
for step in 00001 00192; do
  test ! -e "response_step_$step"
  timeout 850 /root/miniconda3/bin/python -u \
    code_response_v4/audit_ram_source_correctability_response.py \
    --lock "RESPONSE_AUDIT_LOCK_STEP_$step.json" --out "response_step_$step" \
    > "response_step_$step.log" 2>&1
done
