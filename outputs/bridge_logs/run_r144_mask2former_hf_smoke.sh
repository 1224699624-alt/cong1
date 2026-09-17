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
  --limit-train 2 \
  --limit-eval 2 \
  --steps 2 \
  --num-queries 32 \
  --hidden-dim 64 \
  --encoder-layers 1 \
  --decoder-layers 2 \
  --threshold 0.5 \
  --output-dir outputs/r144_mask2former_hf_smoke \
  --metrics-json outputs/analysis/r144_mask2former_hf_smoke_clean_test_v2_metrics.json \
  --manifest-json outputs/analysis/r144_mask2former_hf_smoke_manifest.json \
  --device cuda
