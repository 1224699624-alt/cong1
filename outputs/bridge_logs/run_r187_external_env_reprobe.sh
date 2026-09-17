#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

export PATH="/usr/local/cuda/bin:/home/shenzeyu/miniconda3/bin:${PATH}"
export CUDA_HOME="/usr/local/cuda"
PY="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"

"${PY}" scripts/r187_external_env_reprobe.py \
  2>&1 | tee outputs/bridge_logs/r187_external_env_reprobe.log
