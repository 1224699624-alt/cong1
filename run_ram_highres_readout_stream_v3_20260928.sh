#!/usr/bin/env bash
set -euo pipefail
cd /root/ram_highres_readout_20260928
trap 'echo "$?" > STREAM_EXIT_v3.txt' EXIT
test -f run_v1/NATURAL_Q_probe_calibration.json
test ! -e stream_v3
timeout 3950 /root/miniconda3/bin/python -u \
  code_stream_v3/ram_highres_natural_stream.py \
  --lock HIGHRES_STREAM_EVAL_LOCK_v3.json \
  --out stream_v3 > STREAM_v3.log 2>&1
