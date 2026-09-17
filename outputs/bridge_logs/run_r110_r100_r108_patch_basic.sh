#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES=0
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
$PY scripts/train_patch_disagreement_arbitrator.py \
  --anchor-exp r100_like_r025b_r097_patch_basic_trainval \
  --apply-anchor-exp r100_r095b_r097_patch_basic \
  --candidate-exps r108_dinov3_convnext_tiny_unet \
  --apply-candidate-exps r108_dinov3_convnext_tiny_unet \
  --output-exp r110_r100_r108_patch_basic \
  --checkpoint outputs/patch_arbitrator/r110_r100_r108_patch_basic.pt \
  --metrics-json outputs/analysis/r110_r100_r108_patch_basic_clean_test_v2_metrics.json \
  --feature-mode basic \
  --zone-mode anchor_candidate_disagreement \
  --train-zone-radius 1 \
  --crop-size 96 \
  --crops-per-image 1 \
  --epochs 8 \
  --batch-size 16 \
  --base-channels 20 \
  --lr 0.001 \
  --pos-weight-scale 0.50 \
  --zone-loss-weight 5.0 \
  --boundary-radii 0,1 \
  --prob-thresholds 0.50,0.55,0.60,0.65,0.70,0.75 \
  --edit-margins 0.05,0.10,0.15,0.20 \
  --seed 20260710
