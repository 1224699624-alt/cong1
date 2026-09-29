#!/usr/bin/env bash
# Second matched arm: frozen real O into the same source-head adapter.
set -euo pipefail
cd /root/ram_o_source_head_20260924
trap 'echo "$?" > OHEAD_REAL_EXIT_v1.txt' EXIT
test -f OHEAD_ZERO_v1/FINAL_VALIDATION.json
test ! -e OHEAD_REAL_v1
timeout 1100 /root/miniconda3/bin/python -u \
  code_pair_v1/run_ram_o_source_head_pair.py \
  --lock O_SOURCE_HEAD_RUN_LOCK_v1.json --arm OHEAD_REAL --out OHEAD_REAL_v1 \
  > OHEAD_REAL_v1.log 2>&1
