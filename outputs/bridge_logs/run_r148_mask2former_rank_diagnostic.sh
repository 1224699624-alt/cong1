#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
DEPS=.tmp/r144_hf_deps

PYTHONPATH="$DEPS:${PYTHONPATH:-}" "$PY" scripts/r148_mask2former_rank_diagnostic.py \
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
  --thresholds 0.000001,0.000003,0.000005,0.0000075,0.00001,0.000015,0.00002,0.00005 \
  --topk-factors 0.5,0.75,1.0,1.25,1.5,2.0 \
  --output-json outputs/analysis/r148_mask2former_rank_diagnostic.json \
  --device cuda
