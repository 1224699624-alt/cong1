#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python

"${PY}" scripts/train_timm_instance_separation_segmenter.py \
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
  --weight-decay 1e-4 \
  --boundary-loss-weight 0.12 \
  --sep-loss-weight 0.55 \
  --hover-loss-weight 0.20 \
  --sep-band-kernel 11 \
  --sep-suppress-weights 0.00,0.15,0.30,0.45,0.60 \
  --thresholds 0.45,0.50,0.55,0.60,0.65,0.70,0.75 \
  --hard-case-manifest outputs/analysis/r125_hard_case_curation_manifest/r125_hard_case_curation_manifest.json \
  --hard-case-weight 3.0 \
  --hard-case-epoch-multiplier 1.25 \
  --output-exp r126_hardcase_sampler_instance_sep \
  --checkpoint outputs/timm_instance_sep/r126_hardcase_sampler_instance_sep/best.pt \
  --history-json outputs/timm_instance_sep/r126_hardcase_sampler_instance_sep/history.json \
  --metrics-json outputs/analysis/r126_hardcase_sampler_instance_sep_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r126_hardcase_sampler_instance_sep_original_test_metrics.json \
  --seed 20260726 \
  --device cuda
