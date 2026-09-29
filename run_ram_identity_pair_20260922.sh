#!/usr/bin/env bash
set -euo pipefail
cd /root/ram_identity_causal_20260922
trap 'echo "$?" > PAIR_EXIT.txt' EXIT
test ! -e scrambled
test ! -e canonical
timeout 2100 /root/miniconda3/bin/python -u train_code/train_ram_identity_causal.py --lock DIAGNOSTIC_RUN_LOCK_scrambled.json --out scrambled > scrambled.log 2>&1
timeout 2100 /root/miniconda3/bin/python -u train_code/train_ram_identity_causal.py --lock DIAGNOSTIC_RUN_LOCK_canonical.json --out canonical > canonical.log 2>&1
