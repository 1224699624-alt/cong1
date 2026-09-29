#!/usr/bin/env bash
# Read-only retry after SHA parser bug; no model or training changes.
set -euo pipefail
cd /root/ram_source_correctability_20260924
trap 'echo "$?" > RESPONSE_EXIT_v2.txt' EXIT
test ! -e response_v2
timeout 700 /root/miniconda3/bin/python -u \
  code_response_v2/audit_ram_source_correctability_response.py \
  --lock RESPONSE_AUDIT_LOCK_v2.json --out response_v2 \
  > response_v2.log 2>&1
