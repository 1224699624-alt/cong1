#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python

"$PY" scripts/train_timm_unet_segmenter.py \
  --encoder convnext_tiny.dinov3_lvd1689m \
  --pretrained \
  --img-size 768 \
  --epochs 48 \
  --min-epochs 12 \
  --patience 10 \
  --batch-size 2 \
  --num-workers 4 \
  --decoder-channels 160 \
  --lr 1.5e-4 \
  --weight-decay 1e-4 \
  --boundary-loss-weight 0.12 \
  --distance-loss-weight 0.25 \
  --distance-sigma 5.0 \
  --thresholds 0.40,0.45,0.50,0.55,0.60 \
  --min-component-areas 0,16,32 \
  --strong-xray-aug \
  --grad-accum-steps 2 \
  --ema-decay 0.997 \
  --output-exp r143_highres_medical_recipe_unet \
  --checkpoint outputs/timm_unet/r143_highres_medical_recipe_unet/best.pt \
  --history-json outputs/timm_unet/r143_highres_medical_recipe_unet/history.json \
  --metrics-json outputs/analysis/r143_highres_medical_recipe_unet_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r143_highres_medical_recipe_unet_original_test_metrics.json \
  --seed 20260701 \
  --device cuda
