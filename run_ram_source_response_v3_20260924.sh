#!/usr/bin/env bash
# Full 97-case read-only response with hooks on fixed representative cases.
set -euo pipefail
cd /root/ram_source_correctability_20260924
trap 'echo "$?" > RESPONSE_EXIT_v3.txt' EXIT
test ! -e response_v3
timeout 1850 /root/miniconda3/bin/python -u \
  code_response_v3/audit_ram_source_correctability_response.py \
  --lock RESPONSE_AUDIT_LOCK_v3.json --out response_v3 \
  > response_v3.log 2>&1
