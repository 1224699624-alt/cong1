#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/shenzeyu/workspace/YOLO_SAM_generic_src"; PYTHON="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"; cd "$ROOT"; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"; RESULT="outputs/analysis/r257b_scale_repair_sanity_${STAMP}.json"; CSV="outputs/analysis/r257b_scale_repair_sanity_proposals_${STAMP}.csv"; HEAD="outputs/scale_repair/r257b_scale_repair_sanity/log_scale_head_seed2571_${STAMP}.pt"
"$PYTHON" scripts/train_r257b_scale_repair_comparison.py --variant-root data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1 --output-dir outputs/scale_repair/r257b_scale_repair_sanity --log-scale-checkpoint "$HEAD" --result-json "$RESULT" --proposal-csv "$CSV" --limit-train 32 --limit-val 16 --epochs 1 --batch-size 8 --workers 4 --device cuda
cp "$RESULT" outputs/analysis/r257b_scale_repair_sanity.json; cp "$CSV" outputs/analysis/r257b_scale_repair_sanity_proposals.csv
