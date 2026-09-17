#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
RUN_ID=r182_topology_cldice_smoke

"${PY}" scripts/train_timm_instance_separation_segmenter.py \
  --dataset TSRS_RSNA-Epiphysis \
  --raw-root data/raw \
  --eval-raw-root data/raw_variants \
  --eval-dataset TSRS_RSNA-Epiphysis_clean_test_v2 \
  --control-raw-root data/raw \
  --control-dataset TSRS_RSNA-Epiphysis \
  --encoder convnext_tiny.dinov3_lvd1689m \
  --pretrained \
  --img-size 512 \
  --epochs 8 \
  --min-epochs 4 \
  --patience 4 \
  --batch-size 2 \
  --num-workers 4 \
  --decoder-channels 128 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --boundary-loss-weight 0.16 \
  --sep-loss-weight 0.45 \
  --hover-loss-weight 0.20 \
  --bridge-gap-loss-weight 0.18 \
  --background-loss-weight 0.04 \
  --cldice-loss-weight 0.08 \
  --skeleton-iters 8 \
  --sep-band-kernel 11 \
  --thresholds 0.45,0.50,0.55,0.60,0.65,0.70 \
  --sep-suppress-weights 0.00,0.15,0.30,0.45 \
  --limit-train 16 \
  --limit-val 8 \
  --limit-eval 4 \
  --output-exp "${RUN_ID}" \
  --checkpoint "outputs/timm_instance_sep/${RUN_ID}/best.pt" \
  --history-json "outputs/timm_instance_sep/${RUN_ID}/history.json" \
  --metrics-json "outputs/analysis/${RUN_ID}_clean_test_v2_metrics.json" \
  --control-metrics-json "outputs/analysis/${RUN_ID}_original_test_metrics.json" \
  --pred-root outputs/ablations_variants \
  --seed 2026073182
