#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Exact full-data R260/R258B prior protocol: 875 train, 96 original-val.
# No selection manifest is passed and test/clean-test-v2 are unsupported.
python scripts/train_r258b_prediction_relation_prior.py \
  --output-dir outputs/pair_prior/r318_full_r260_prior \
  --result-json outputs/analysis/r318_full_r260_prior.json \
  --pair-manifest-csv outputs/analysis/r318_full_r260_prior_pairs.csv \
  --workers 6

python scripts/build_r259_frozen_prior_maps.py \
  --r258b-result outputs/analysis/r318_full_r260_prior.json \
  --checkpoint-dir outputs/pair_prior/r318_full_r260_prior \
  --output-root outputs/priors/r318_full_r260_relation

sha256sum \
  outputs/analysis/r318_full_r260_prior.json \
  outputs/analysis/r318_full_r260_prior_pairs.csv \
  outputs/priors/r318_full_r260_relation/manifest.json \
  > outputs/analysis/r318_full_r260_prior.sha256
