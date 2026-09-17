#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src
OUT=outputs/ram_w600/r337_frozen_prior_backprop
mkdir -p "$OUT" outputs/bridge_logs
/root/miniconda3/bin/python -u scripts/train_r337_ram_frozen_prior_backprop.py \
  --dataset-root data/remote_variants/RAM-W600 \
  --baseline-checkpoint outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth \
  --prior-checkpoint outputs/ram_w600/r330_pair_surface_volume_prior/surface003_volume010_best.pth \
  --output "$OUT" --epochs 30 --learning-rate 7e-6 --prior-weight 0.08 \
  --benefit-margin 0.002 --max-delta 0.75 \
  2>&1 | tee outputs/bridge_logs/r337_frozen_prior_backprop.log
