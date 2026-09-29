#!/usr/bin/env bash
# Frozen C0/O conditional-information audit only; no segmentation updates.
set -euo pipefail
cd /root/ram_prior_info_20260924
trap 'echo "$?" > PRIOR_INFORMATION_EXIT_v1.txt' EXIT
test ! -e audit_v1
timeout 2700 /root/miniconda3/bin/python -u \
  code_prior_v1/audit_ram_o_conditional_h0.py \
  --lock PRIOR_INFORMATION_LOCK_v1.json --out audit_v1 \
  > audit_v1.log 2>&1
