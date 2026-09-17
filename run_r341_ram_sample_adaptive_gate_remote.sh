#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src
OUT=outputs/ram_w600/r341_sample_adaptive_gate
mkdir -p "$OUT" outputs/bridge_logs
/root/miniconda3/bin/python -u scripts/train_r341_ram_sample_adaptive_gate.py \
 --dataset-root data/remote_variants/RAM-W600 \
 --baseline-checkpoint outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth \
 --prior-checkpoint outputs/ram_w600/r339_prior_from_scratch/instance_prior_best.pth \
 --r325-cache outputs/ram_w600/r332_joint_iterative_refinement/r325_official_cache.json \
 --r332-result outputs/ram_w600/r332_joint_iterative_refinement/steps_3/result.json \
 --output "$OUT" --epochs 30 --warmup-epochs 4 \
 2>&1 | tee outputs/bridge_logs/r341_sample_adaptive_gate.log
