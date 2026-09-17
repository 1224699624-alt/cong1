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
  --hf-arch upernet \
  --model-name openmmlab/upernet-convnext-tiny \
  --extra-pythonpath .tmp/r144_hf_deps \
  --img-size 512 \
  --epochs 6 \
  --min-epochs 3 \
  --patience 3 \
  --batch-size 2 \
  --num-workers 4 \
  --lr 6e-5 \
  --boundary-loss-weight 0.08 \
  --distance-loss-weight 0.04 \
  --strong-xray-aug \
  --ema-decay 0.0 \
  --limit-train 16 \
  --limit-val 8 \
  --limit-eval 4 \
  --thresholds 0.30,0.40,0.50,0.60,0.70 \
  --min-component-areas 0,8,16 \
  --output-exp r160_hf_upernet_convnext_smoke \
  --checkpoint outputs/hf_semantic/r160_hf_upernet_convnext_smoke/best.pt \
  --history-json outputs/hf_semantic/r160_hf_upernet_convnext_smoke/history.json \
  --metrics-json outputs/analysis/r160_hf_upernet_convnext_smoke_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r160_hf_upernet_convnext_smoke_original_test_metrics.json \
  --seed 20260702 \
  --device cuda
