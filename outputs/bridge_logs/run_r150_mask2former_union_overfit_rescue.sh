#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
DEPS=.tmp/r144_hf_deps

PYTHONPATH="$DEPS:${PYTHONPATH:-}" "$PY" scripts/r149_mask2former_overfit_rescue.py \
  --hf-deps "$DEPS" \
  --img-size 384 \
  --limit-train 1 \
  --limit-val 2 \
  --limit-eval 2 \
  --steps 900 \
  --eval-every 100 \
  --num-queries 64 \
  --hidden-dim 128 \
  --encoder-layers 2 \
  --decoder-layers 4 \
  --lr 2e-4 \
  --target-mode union \
  --class-weight 1.0 \
  --mask-weight 12.0 \
  --dice-weight 12.0 \
  --no-object-weight 0.05 \
  --thresholds 0.05,0.1,0.15,0.2,0.25,0.35,0.5,0.65 \
  --output-json outputs/analysis/r150_mask2former_union_overfit_rescue_metrics.json \
  --checkpoint outputs/r150_mask2former_union_overfit_rescue/model.pt \
  --seed 20260702 \
  --device cuda
