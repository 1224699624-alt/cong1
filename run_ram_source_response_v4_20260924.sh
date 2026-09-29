#!/usr/bin/env bash
# Full read-only response; exact H0 counts with sampled background quantiles.
set -euo pipefail
cd /root/ram_source_correctability_20260924
trap 'echo "$?" > RESPONSE_EXIT_v4.txt' EXIT
test ! -e response_v4
timeout 2150 /root/miniconda3/bin/python -u \
  code_response_v4/audit_ram_source_correctability_response.py \
  --lock RESPONSE_AUDIT_LOCK_v4.json --out response_v4 \
  > response_v4.log 2>&1
