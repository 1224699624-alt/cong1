#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python

"${PY}" scripts/train_set_instance_segmenter.py \
  --dataset TSRS_RSNA-Epiphysis \
  --raw-root data/raw \
  --eval-raw-root data/raw_variants \
  --eval-dataset TSRS_RSNA-Epiphysis_clean_test_v2 \
  --control-dataset TSRS_RSNA-Epiphysis \
  --img-size 512 \
  --num-queries 32 \
  --epochs 40 \
  --min-epochs 10 \
  --patience 8 \
  --batch-size 4 \
  --num-workers 4 \
  --base-channels 24 \
  --lr 7e-4 \
  --weight-decay 1e-4 \
  --binary-loss-weight 0.55 \
  --empty-loss-weight 0.02 \
  --overlap-loss-weight 0.03 \
  --thresholds 0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70 \
  --slot-blends 0.75,0.90,1.00 \
  --output-exp r129_set_instance_segmenter \
  --checkpoint outputs/set_instance/r129_set_instance_segmenter/best.pt \
  --history-json outputs/set_instance/r129_set_instance_segmenter/history.json \
  --metrics-json outputs/analysis/r129_set_instance_segmenter_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r129_set_instance_segmenter_original_test_metrics.json \
  --seed 20260729 \
  --device cuda
