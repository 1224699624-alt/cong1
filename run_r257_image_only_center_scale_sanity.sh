#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shenzeyu/workspace/YOLO_SAM_generic_src"
PYTHON="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT="outputs/analysis/r257_image_only_center_scale_sanity_${STAMP}.json"
PROPOSALS="outputs/analysis/r257_image_only_center_scale_sanity_proposals_${STAMP}.csv"
"$PYTHON" scripts/train_r257_image_only_center_scale_detector.py \
  --variant-root data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1 \
  --output-dir outputs/center_detector/r257_image_only_center_scale_sanity \
  --result-json "$RESULT" --proposal-csv "$PROPOSALS" \
  --limit-train 32 --limit-val 16 --epochs 1 --batch-size 8 --workers 4 --device cuda
cp "$RESULT" outputs/analysis/r257_image_only_center_scale_sanity.json
cp "$PROPOSALS" outputs/analysis/r257_image_only_center_scale_sanity_proposals.csv

