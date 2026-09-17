#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"; PYTHON="/root/miniconda3/bin/python"; cd "$ROOT"; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"; RESULT="outputs/analysis/r258b_prediction_relation_prior_full_${STAMP}.json"; PAIRS="outputs/analysis/r258b_prediction_relation_prior_pairs_full_${STAMP}.csv"; OUT="outputs/pair_prior/r258b_prediction_relation_prior_full_${STAMP}"
"$PYTHON" scripts/test_r258b_prediction_relation_prior.py
"$PYTHON" scripts/train_r258b_prediction_relation_prior.py --output-dir "$OUT" --result-json "$RESULT" --pair-manifest-csv "$PAIRS" --crop-size 128 --epochs 4 --batch-size 32 --workers 6 --lr 0.0005 --weight-decay 0.0001 --crop-margin-scales 1.35 --match-tolerance-scale 0.60 --relative-pair-limit 0.80 --proposal-distance-limit 4.0 --relative-close-threshold 0.20 --max-pairs-per-case 32 --seed 2581 --train-seeds 2581,2582,2583 --device cuda
cp "$RESULT" outputs/analysis/r258b_prediction_relation_prior_full.json; cp "$PAIRS" outputs/analysis/r258b_prediction_relation_prior_pairs_full.csv
