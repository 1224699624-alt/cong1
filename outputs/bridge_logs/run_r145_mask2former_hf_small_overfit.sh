#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
DEPS=.tmp/r144_hf_deps

if [ ! -d "$DEPS/transformers" ]; then
  mkdir -p "$DEPS"
  "$PY" -m pip install --quiet --target "$DEPS" \
    "transformers==4.53.0" "safetensors" "huggingface_hub"
fi

PYTHONPATH="$DEPS:${PYTHONPATH:-}" "$PY" scripts/r144_mask2former_hf_smoke.py \
  --hf-deps "$DEPS" \
  --img-size 256 \
  --limit-train 16 \
  --limit-val 4 \
  --limit-eval 4 \
  --steps 120 \
  --num-queries 48 \
  --hidden-dim 96 \
  --encoder-layers 2 \
  --decoder-layers 3 \
  --lr 1e-4 \
  --threshold 0.35 \
  --output-dir outputs/r145_mask2former_hf_small_overfit \
  --metrics-json outputs/analysis/r145_mask2former_hf_small_overfit_clean_test_v2_metrics.json \
  --train-metrics-json outputs/analysis/r145_mask2former_hf_small_overfit_train_metrics.json \
  --val-metrics-json outputs/analysis/r145_mask2former_hf_small_overfit_val_metrics.json \
  --manifest-json outputs/analysis/r145_mask2former_hf_small_overfit_manifest.json \
  --seed 20260702 \
  --device cuda
