#!/usr/bin/env bash
# Read-only H0/O audit with case-null-SHA manifest digest protocol.
set -euo pipefail
cd /root/ram_prior_info_20260924
trap 'echo "$?" > PRIOR_INFORMATION_EXIT_v2.txt' EXIT
test ! -e audit_v2
timeout 2700 /root/miniconda3/bin/python -u \
  code_prior_v1/audit_ram_o_conditional_h0.py \
  --lock PRIOR_INFORMATION_LOCK_v2.json --out audit_v2 \
  > audit_v2.log 2>&1
