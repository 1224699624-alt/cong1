#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
ROOT=/root/ram_source_probe_20260922
PY=/root/miniconda3/bin/python
cd "$ROOT/pilot_code"
for ARM in pilot_reference pilot_source_aux; do
  timeout 1800 "$PY" -u whitebox_ram_diagnosis.py \
    --lock "$ROOT/DIAGNOSTIC_RUN_LOCK_${ARM}.json" \
    --out "$ROOT/$ARM" > "$ROOT/${ARM}.log" 2>&1
  "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="COMPLETE" and r["updates"]==100' "$ROOT/$ARM/RESULT.json"
done
printf 'COMPLETE\n' > "$ROOT/PAIR_STATUS.txt"
