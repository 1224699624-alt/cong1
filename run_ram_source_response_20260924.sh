#!/usr/bin/env bash
# Read-only fixed-H0 response audit after all four matched endpoints complete.
set -euo pipefail
cd /root/ram_source_correctability_20260924
trap 'echo "$?" > RESPONSE_EXIT_v1.txt' EXIT
test ! -e response_v1
timeout 700 /root/miniconda3/bin/python -u \
  code_response_v1/audit_ram_source_correctability_response.py \
  --lock RESPONSE_AUDIT_LOCK_v1.json --out response_v1 \
  > response_v1.log 2>&1
