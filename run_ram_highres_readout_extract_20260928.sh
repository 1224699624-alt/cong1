#!/usr/bin/env bash
set -euo pipefail
cd /root/ram_highres_readout_20260928
trap 'echo "$?" > EXTRACT_EXIT_v1.txt' EXIT
test ! -e run_v1
timeout 2500 /root/miniconda3/bin/python -u \
  code_v1/run_ram_highres_readout.py \
  --mode extract \
  --lock HIGHRES_EXTRACTION_LOCK_v1.json \
  --out run_v1 > EXTRACT_v1.log 2>&1
