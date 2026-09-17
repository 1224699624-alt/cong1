#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python

"$PY" scripts/train_hf_segformer_segmenter.py \
  --dataset TSRS_RSNA-Epiphysis \
  --raw-root data/raw \
  --eval-raw-root data/raw_variants \
  --eval-dataset TSRS_RSNA-Epiphysis_clean_test_v2 \
  --control-dataset TSRS_RSNA-Epiphysis \
  --model-name nvidia/segformer-b0-finetuned-ade-512-512 \
  --extra-pythonpath .tmp/r144_hf_deps \
  --img-size 512 \
  --epochs 60 \
  --min-epochs 20 \
  --patience 15 \
  --batch-size 1 \
  --num-workers 2 \
  --lr 1e-4 \
  --boundary-loss-weight 0.04 \
  --distance-loss-weight 0.02 \
  --limit-train 1 \
  --limit-val 1 \
  --thresholds 0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80 \
  --min-component-areas 0 \
  --output-exp r158_hf_segformer_one_image_overfit \
  --checkpoint outputs/hf_segformer/r158_hf_segformer_one_image_overfit/best.pt \
  --history-json outputs/hf_segformer/r158_hf_segformer_one_image_overfit/history.json \
  --metrics-json outputs/analysis/r158_hf_segformer_one_image_overfit_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r158_hf_segformer_one_image_overfit_original_test_metrics.json \
  --skip-final-eval \
  --seed 20260702 \
  --device cuda
