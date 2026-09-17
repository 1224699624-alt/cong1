#!/usr/bin/env bash
set -euo pipefail
export R255_PROJECT_ROOT_OVERRIDE="/root/autodl-tmp/YOLO_SAM_generic_src"
export R255_PYTHON="/root/miniconda3/bin/python"
export R255_BIN="/root/miniconda3/bin"
export R255_GPU="0"
exec bash "$R255_PROJECT_ROOT_OVERRIDE/outputs/bridge_logs/run_r255_close_gap_nnunet_sanity.sh"
