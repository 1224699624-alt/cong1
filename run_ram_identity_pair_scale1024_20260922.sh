#!/usr/bin/env bash
set -euo pipefail
cd /root/ram_identity_causal_20260922
trap 'echo "$?" > PAIR_SCALE1024_EXIT.txt' EXIT
test ! -e scrambled_scale1024
test ! -e canonical_scale1024
timeout 2100 /root/miniconda3/bin/python -u scale1024_code/train_ram_identity_causal.py --lock DIAGNOSTIC_RUN_LOCK_scrambled_scale1024.json --out scrambled_scale1024 > scrambled_scale1024.log 2>&1
timeout 2100 /root/miniconda3/bin/python -u scale1024_code/train_ram_identity_causal.py --lock DIAGNOSTIC_RUN_LOCK_canonical_scale1024.json --out canonical_scale1024 > canonical_scale1024.log 2>&1
