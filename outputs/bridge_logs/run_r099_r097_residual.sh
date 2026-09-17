#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES=0
/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python scripts/train_anchor_pixel_residual.py \
  --anchor-exp r025b_anchor_local_pixel_residual_sam_hqsam \
  --apply-anchor-exp r095b_r036_patch_basic \
  --candidate-exps r097_timm_unet_convnext_tiny \
  --apply-candidate-exps r097_timm_unet_convnext_tiny \
  --output-exp r099_r095b_r097_local_stats_residual \
  --checkpoint outputs/pixel_residual/r099_r095b_r097_local_stats_residual.pt \
  --metrics-json outputs/analysis/r099_r095b_r097_local_stats_residual_clean_test_v2_metrics.json \
  --feature-mode local_stats \
  --zone-mode anchor_candidate_disagreement \
  --train-zone-radius 1 \
  --samples-per-image 1536 \
  --hidden 48 \
  --epochs 10 \
  --batch-size 65536 \
  --lr 0.0015 \
  --pos-weight-scale 0.60 \
  --boundary-radii 0,1 \
  --prob-thresholds 0.50,0.55,0.60,0.65,0.70,0.75 \
  --edit-margins 0.05,0.10,0.15,0.20 \
  --seed 20260699
