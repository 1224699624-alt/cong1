#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES=0
for split in train val; do
  /home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python scripts/apply_patch_disagreement_arbitrator.py \
    --checkpoint outputs/patch_arbitrator/r100_r095b_r097_patch_basic.pt \
    --dataset TSRS_RSNA-Epiphysis \
    --raw-root data/raw \
    --split ${split} \
    --anchor-exp r025b_anchor_local_pixel_residual_sam_hqsam \
    --candidate-exps r097_timm_unet_convnext_tiny \
    --output-exp r100_like_r025b_r097_patch_basic_trainval \
    --metrics-json outputs/analysis/r100_like_r025b_r097_patch_basic_${split}_metrics.json \
    --feature-mode basic \
    --zone-mode anchor_candidate_disagreement \
    --radius 1 \
    --threshold 0.50 \
    --margin 0.10
done
