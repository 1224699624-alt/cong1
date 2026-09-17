#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
DEPS=.tmp/r144_hf_deps

PYTHONPATH="$DEPS:${PYTHONPATH:-}" "$PY" scripts/r149_mask2former_overfit_rescue.py \
  --hf-deps "$DEPS" \
  --img-size 256 \
  --limit-train 1 \
  --limit-val 2 \
  --limit-eval 2 \
  --steps 500 \
  --eval-every 50 \
  --num-queries 48 \
  --hidden-dim 96 \
  --encoder-layers 2 \
  --decoder-layers 3 \
  --lr 3e-4 \
  --target-mode union \
  --thresholds 0.05,0.1,0.2,0.35,0.5 \
  --output-json outputs/analysis/r149_mask2former_union_overfit_rescue_metrics.json \
  --checkpoint outputs/r149_mask2former_union_overfit_rescue/model.pt \
  --seed 20260702 \
  --device cuda
