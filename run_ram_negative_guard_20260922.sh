#!/usr/bin/env bash
# Reproduction entry point. Fails if output exists; does not overwrite a run.
set -euo pipefail
cd /root/ram_negative_guard_20260922
test ! -e pilot_guard
timeout 1800 /root/miniconda3/bin/python -u pilot_code/whitebox_ram_diagnosis.py \
  --lock DIAGNOSTIC_RUN_LOCK_pilot_guard.json --out pilot_guard
