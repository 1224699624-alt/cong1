#!/usr/bin/env bash
# Read-only full14 response of final-H0 target on fixed RAM train/val cases.
set -euo pipefail
cd /root/ram_h0_final_target_20260924
trap 'echo "$?" > FINAL_RESPONSE_EXIT_v1.txt' EXIT
test -f FIN_F0S1_v2/FINAL_VALIDATION.json
test ! -e response_v1
timeout 600 /root/miniconda3/bin/python -u \
  code_final_response_v1/audit_ram_h0_final_target_response.py \
  --lock FINAL_RESPONSE_LOCK_v1.json --out response_v1 \
  > response_v1.log 2>&1
