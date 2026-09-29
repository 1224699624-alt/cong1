#!/usr/bin/env bash
# First matched control: same zero-init source-head adapter with O=0.
set -euo pipefail
cd /root/ram_o_source_head_20260924
trap 'echo "$?" > OHEAD_ZERO_EXIT_v1.txt' EXIT
test ! -e OHEAD_ZERO_v1
timeout 1100 /root/miniconda3/bin/python -u \
  code_pair_v1/run_ram_o_source_head_pair.py \
  --lock O_SOURCE_HEAD_RUN_LOCK_v1.json --arm OHEAD_ZERO --out OHEAD_ZERO_v1 \
  > OHEAD_ZERO_v1.log 2>&1
