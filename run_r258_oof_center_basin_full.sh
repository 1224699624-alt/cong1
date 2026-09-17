#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shenzeyu/workspace/YOLO_SAM_generic_src"; PYTHON="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"; cd "$ROOT"; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"; OUT="outputs/oof_proposals/r258_center_basin_oof_full_${STAMP}"; RESULT="outputs/analysis/r258_oof_center_basin_full_${STAMP}.json"; CSV="outputs/analysis/r258_oof_center_basin_full_proposals_${STAMP}.csv"; MANIFEST="outputs/analysis/r258_oof_center_basin_full_folds_${STAMP}.csv"; INFERENCE="outputs/analysis/r258_oof_center_basin_full_inference_${STAMP}.csv"
"$PYTHON" scripts/test_r258_oof_center_basin_proposals.py
"$PYTHON" scripts/train_r258_oof_center_basin_proposals.py --variant-root data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1 --metadata outputs/metadata/r256/filtered_train.csv --output-dir "$OUT" --result-json "$RESULT" --proposal-csv "$CSV" --fold-manifest-csv "$MANIFEST" --inference-manifest-csv "$INFERENCE" --num-folds 5 --epochs 8 --batch-size 8 --workers 6 --device cuda
cp "$RESULT" outputs/analysis/r258_oof_center_basin_full.json; cp "$CSV" outputs/analysis/r258_oof_center_basin_full_proposals.csv; cp "$MANIFEST" outputs/analysis/r258_oof_center_basin_full_folds.csv; cp "$INFERENCE" outputs/analysis/r258_oof_center_basin_full_inference.csv
