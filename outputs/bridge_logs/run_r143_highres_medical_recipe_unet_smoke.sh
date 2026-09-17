#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python

"$PY" scripts/train_timm_unet_segmenter.py \
  --encoder convnext_tiny.dinov3_lvd1689m \
  --pretrained \
  --img-size 256 \
  --epochs 1 \
  --min-epochs 1 \
  --patience 1 \
  --batch-size 1 \
  --num-workers 0 \
  --decoder-channels 64 \
  --lr 2e-4 \
  --boundary-loss-weight 0.10 \
  --distance-loss-weight 0.20 \
  --distance-sigma 5.0 \
  --thresholds 0.45,0.50 \
  --min-component-areas 0,16 \
  --strong-xray-aug \
  --grad-accum-steps 1 \
  --ema-decay 0.0 \
  --tta-flips \
  --limit-train 4 \
  --limit-val 2 \
  --limit-eval 2 \
  --output-exp r143_highres_medical_recipe_unet_smoke \
  --checkpoint outputs/timm_unet/r143_highres_medical_recipe_unet_smoke/best.pt \
  --history-json outputs/timm_unet/r143_highres_medical_recipe_unet_smoke/history.json \
  --metrics-json outputs/analysis/r143_highres_medical_recipe_unet_smoke_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r143_highres_medical_recipe_unet_smoke_original_test_metrics.json \
  --seed 20260701 \
  --device cuda
