#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"; PYTHON="/root/miniconda3/bin/python"; cd "$ROOT"; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"; RESULT="outputs/analysis/r258b_prediction_relation_prior_sanity_${STAMP}.json"; PAIRS="outputs/analysis/r258b_prediction_relation_prior_pairs_sanity_${STAMP}.csv"; OUT="outputs/pair_prior/r258b_prediction_relation_prior_sanity_${STAMP}"
"$PYTHON" scripts/test_r258b_prediction_relation_prior.py
"$PYTHON" scripts/train_r258b_prediction_relation_prior.py --output-dir "$OUT" --result-json "$RESULT" --pair-manifest-csv "$PAIRS" --limit-train 40 --limit-val 16 --epochs 1 --train-seeds 2581 --workers 2 --device cuda
cp "$RESULT" outputs/analysis/r258b_prediction_relation_prior_sanity.json; cp "$PAIRS" outputs/analysis/r258b_prediction_relation_prior_pairs_sanity.csv
