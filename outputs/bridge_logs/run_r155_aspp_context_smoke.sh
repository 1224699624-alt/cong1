#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python

"$PY" scripts/train_timm_aspp_segmenter.py \
  --dataset TSRS_RSNA-Epiphysis \
  --raw-root data/raw \
  --eval-raw-root data/raw_variants \
  --eval-dataset TSRS_RSNA-Epiphysis_clean_test_v2 \
  --control-dataset TSRS_RSNA-Epiphysis \
  --encoder convnext_tiny \
  --pretrained \
  --img-size 512 \
  --epochs 6 \
  --min-epochs 3 \
  --patience 3 \
  --batch-size 2 \
  --num-workers 4 \
  --decoder-channels 128 \
  --aspp-rates 1,3,6,9 \
  --lr 3e-4 \
  --boundary-loss-weight 0.12 \
  --distance-loss-weight 0.08 \
  --strong-xray-aug \
  --ema-decay 0.995 \
  --limit-train 16 \
  --limit-val 8 \
  --limit-eval 4 \
  --thresholds 0.35,0.45,0.55,0.65 \
  --min-component-areas 0,8,16 \
  --output-exp r155_aspp_context_smoke \
  --checkpoint outputs/timm_aspp/r155_aspp_context_smoke/best.pt \
  --history-json outputs/timm_aspp/r155_aspp_context_smoke/history.json \
  --metrics-json outputs/analysis/r155_aspp_context_smoke_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r155_aspp_context_smoke_original_test_metrics.json \
  --seed 20260702 \
  --device cuda
