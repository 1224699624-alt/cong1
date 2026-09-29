#!/usr/bin/env bash
# Same read-only endpoint audit with absolute original schedule path in lock.
set -euo pipefail
cd /root/ram_h0_final_target_20260924
trap 'echo "$?" > FINAL_RESPONSE_EXIT_v2.txt' EXIT
test -f FIN_F0S1_v2/FINAL_VALIDATION.json
test ! -e response_v2
timeout 600 /root/miniconda3/bin/python -u \
  code_final_response_v1/audit_ram_h0_final_target_response.py \
  --lock FINAL_RESPONSE_LOCK_v2.json --out response_v2 \
  > response_v2.log 2>&1
