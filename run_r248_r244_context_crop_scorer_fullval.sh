#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

mkdir -p outputs/analysis outputs/bridge_logs

/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python scripts/train_r248_r244_context_crop_scorer.py \
  --dataset TSRS_RSNA-Epiphysis \
  --raw-root data/raw \
  --ablations-root outputs/ablations_variants \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --split val \
  --dist-percentiles 6,8,10,12 \
  --max-candidates-generated 64 \
  --max-components-per-image 5 \
  --epochs 8 \
  --batch-size 64 \
  --grouped-cv-folds 5 \
  --flush-every 10 \
  --candidate-snapshot-csv outputs/analysis/r248_r244_context_crop_scorer_fullval_candidates_snapshot.csv \
  --progress-json outputs/analysis/r248_r244_context_crop_scorer_fullval_progress.json \
  --output-json outputs/analysis/r248_r244_context_crop_scorer_fullval.json \
  --grid-csv outputs/analysis/r248_r244_context_crop_scorer_fullval_threshold_grid.csv \
  --probed-csv outputs/analysis/r248_r244_context_crop_scorer_fullval_probed_rows.csv \
  2>&1 | tee outputs/bridge_logs/r248_r244_context_crop_scorer_fullval.log
