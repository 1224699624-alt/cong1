#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
mkdir -p outputs/analysis outputs/models outputs/bridge_logs

stamp="$(date +%Y%m%d_%H%M%S)"
log="outputs/bridge_logs/r251_reverse_background_channel_smoke_${stamp}.log"
set +e
/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python scripts/train_r251_reverse_background_channel.py \
  --dataset TSRS_RSNA-Epiphysis \
  --raw-root data/raw \
  --ablations-root outputs/ablations_variants \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --train-split train \
  --eval-split train \
  --limit-train 8 \
  --limit-eval 8 \
  --samples-per-image 4096 \
  --positive-samples-per-image 2048 \
  --eval-samples-per-image 8192 \
  --output-json outputs/analysis/r251_reverse_background_channel_smoke.json \
  --model-path outputs/models/r251_reverse_background_channel_smoke.pkl \
  2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
cp "$log" outputs/bridge_logs/r251_reverse_background_channel_smoke.log
exit "$status"
