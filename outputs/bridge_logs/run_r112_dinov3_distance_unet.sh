#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES=0
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
$PY scripts/train_timm_unet_segmenter.py \
  --encoder convnext_tiny.dinov3_lvd1689m \
  --pretrained \
  --img-size 512 \
  --epochs 36 \
  --min-epochs 10 \
  --patience 8 \
  --batch-size 4 \
  --num-workers 4 \
  --decoder-channels 128 \
  --lr 3e-4 \
  --boundary-loss-weight 0.08 \
  --distance-loss-weight 0.20 \
  --distance-sigma 4.0 \
  --thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
  --output-exp r112_dinov3_distance_unet \
  --checkpoint outputs/timm_unet/r112_dinov3_distance_unet/best.pt \
  --history-json outputs/timm_unet/r112_dinov3_distance_unet/history.json \
  --metrics-json outputs/analysis/r112_dinov3_distance_unet_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r112_dinov3_distance_unet_original_test_metrics.json \
  --seed 20260628 \
  --device cuda
