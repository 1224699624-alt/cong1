#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES=0
/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python scripts/train_timm_unet_segmenter.py \
  --encoder convnext_tiny \
  --img-size 512 \
  --epochs 36 \
  --min-epochs 10 \
  --patience 8 \
  --batch-size 4 \
  --num-workers 6 \
  --decoder-channels 128 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --boundary-loss-weight 0.08 \
  --thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
  --output-exp r097_timm_unet_convnext_tiny \
  --checkpoint outputs/timm_unet/r097_timm_unet_convnext_tiny/best.pt \
  --history-json outputs/timm_unet/r097_timm_unet_convnext_tiny/history.json \
  --metrics-json outputs/analysis/r097_timm_unet_convnext_tiny_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r097_timm_unet_convnext_tiny_original_test_metrics.json \
  --device cuda
