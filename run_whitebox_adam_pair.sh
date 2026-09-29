#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
ROOT=/root/whitebox_adam_20260922
PY=/root/miniconda3/bin/python
cd "$ROOT/code"
for ARM in fresh_adam restored_adam; do
  timeout 3000 "$PY" -u whitebox_ram_diagnosis.py \
    --lock "$ROOT/DIAGNOSTIC_RUN_LOCK_${ARM}.json" \
    --out "$ROOT/$ARM" > "$ROOT/${ARM}.log" 2>&1
  "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="COMPLETE" and r["updates"]==100' "$ROOT/$ARM/RESULT.json"
done
printf 'COMPLETE\n' > "$ROOT/PAIR_STATUS.txt"
