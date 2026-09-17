#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shenzeyu/workspace/YOLO_SAM_generic_src"
PYTHON="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RESULT="outputs/analysis/r256_scale_invariant_pair_prior_fullval_${STAMP}.json"
"$PYTHON" scripts/train_r256_scale_invariant_pair_prior.py \
  --variant-root data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1 \
  --train-metadata outputs/metadata/r256/filtered_train.csv \
  --val-metadata outputs/metadata/r256/filtered_val.csv \
  --output-dir outputs/pair_prior/r256_scale_invariant_pair_prior_fullval \
  --result-json "$RESULT" \
  --epochs 4 --train-seeds 256,257,258 --batch-size 128 --workers 8 \
  --device cuda
cp "$RESULT" outputs/analysis/r256_scale_invariant_pair_prior_fullval.json
