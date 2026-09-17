#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
DEPS=.tmp/r144_hf_deps

PYTHONPATH="$DEPS:${PYTHONPATH:-}" "$PY" scripts/r147_mask2former_logits_threshold_diagnostic.py \
  --hf-deps "$DEPS" \
  --img-size 256 \
  --limit-train 16 \
  --limit-val 4 \
  --limit-eval 4 \
  --num-queries 48 \
  --hidden-dim 96 \
  --encoder-layers 2 \
  --decoder-layers 3 \
  --checkpoint outputs/r146_mask2former_hf_maskonly_readout/model.pt \
  --thresholds 0.001,0.003,0.005,0.01,0.02,0.05,0.1,0.2,0.35 \
  --output-json outputs/analysis/r147_mask2former_logits_threshold_diagnostic.json \
  --device cuda
