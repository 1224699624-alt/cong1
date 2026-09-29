#!/usr/bin/env bash
set -euo pipefail
cd /root/ram_highres_readout_20260928
trap 'echo "$?" > FIT_EVAL_EXIT_v1.txt' EXIT
test -f run_v1/SAMPLE_EXTRACTION_RESULT.json
test ! -f run_v1/PROBE_FIT_RESULT.json
timeout 6760 /root/miniconda3/bin/python -u \
  code_v1/run_ram_highres_readout.py \
  --mode fit_eval \
  --lock HIGHRES_READOUT_RUN_LOCK_v1.json \
  --out run_v1 > FIT_EVAL_v1.log 2>&1
