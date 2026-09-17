#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES=0
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python

$PY scripts/apply_timm_unet_segmenter.py \
  --checkpoint outputs/timm_unet/r112_dinov3_distance_unet/best.pt \
  --raw-root data/raw \
  --dataset TSRS_RSNA-Epiphysis \
  --split train \
  --output-exp r112_dinov3_distance_unet \
  --metrics-json outputs/analysis/r112_dinov3_distance_unet_train_metrics.json \
  --device cuda

$PY scripts/apply_timm_unet_segmenter.py \
  --checkpoint outputs/timm_unet/r112_dinov3_distance_unet/best.pt \
  --raw-root data/raw \
  --dataset TSRS_RSNA-Epiphysis \
  --split val \
  --output-exp r112_dinov3_distance_unet \
  --metrics-json outputs/analysis/r112_dinov3_distance_unet_val_metrics.json \
  --device cuda

$PY scripts/train_patch_disagreement_arbitrator.py \
  --anchor-exp r100_like_r025b_r097_patch_basic_trainval \
  --apply-anchor-exp r100_r095b_r097_patch_basic \
  --candidate-exps r112_dinov3_distance_unet \
  --apply-candidate-exps r112_dinov3_distance_unet \
  --output-exp r114_r100_r112_patch_basic \
  --checkpoint outputs/patch_arbitrator/r114_r100_r112_patch_basic.pt \
  --metrics-json outputs/analysis/r114_r100_r112_patch_basic_clean_test_v2_metrics.json \
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
  --seed 20260714
