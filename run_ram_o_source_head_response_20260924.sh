#!/usr/bin/env bash
# Read-only fixed-H0 response for matched O=zero and real-O endpoints.
set -euo pipefail
cd /root/ram_o_source_head_20260924
trap 'echo "$?" > OHEAD_RESPONSE_EXIT_v1.txt' EXIT
test -f OHEAD_ZERO_v1/FINAL_VALIDATION.json
test -f OHEAD_REAL_v1/FINAL_VALIDATION.json
test ! -e response_v1
timeout 700 /root/miniconda3/bin/python -u \
  code_response_v1/audit_ram_o_source_head_response.py \
  --lock OHEAD_RESPONSE_LOCK_v1.json --out response_v1 \
  > response_v1.log 2>&1
