#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python

"${PY}" scripts/train_timm_set_instance_segmenter.py \
  --encoder convnext_tiny.dinov3_lvd1689m \
  --pretrained \
  --img-size 512 \
  --num-queries 32 \
  --epochs 36 \
  --min-epochs 10 \
  --patience 8 \
  --batch-size 4 \
  --num-workers 4 \
  --decoder-channels 128 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --binary-loss-weight 0.45 \
  --object-loss-weight 0.35 \
  --no-object-loss-weight 0.08 \
  --empty-mask-loss-weight 0.02 \
  --overlap-loss-weight 0.04 \
  --thresholds 0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
  --slot-blends 0.70,0.85,1.00 \
  --output-exp r132_dinov3_query_noobject \
  --checkpoint outputs/timm_set_instance/r132_dinov3_query_noobject/best.pt \
  --history-json outputs/timm_set_instance/r132_dinov3_query_noobject/history.json \
  --metrics-json outputs/analysis/r132_dinov3_query_noobject_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r132_dinov3_query_noobject_original_test_metrics.json \
  --seed 20260732 \
  --device cuda
