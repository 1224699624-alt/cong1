#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PYTHON="/root/miniconda3/bin/python"
OUTPUT="outputs/experiments/r321_promptfree_medsam_adapter_official448_seed3201"
LOG="outputs/bridge_logs/r321_promptfree_medsam_adapter_official448_seed3201.log"

cd "$ROOT"
mkdir -p outputs/bridge_logs outputs/experiments

exec "$PYTHON" scripts/run_r321_paired_promptfree_medsam_adapter.py \
  --checkpoint /root/autodl-tmp/external/checkpoints/medsam_vit_b.pth \
  --medsam-repo /root/autodl-tmp/external/MedSAM \
  --output-root "$OUTPUT" \
  --image-size 448 \
  --adapter-ratio 0.25 \
  --batch-size 16 \
  --epochs 300 \
  --min-epochs 300 \
  --patience 300 \
  --lr 1e-4 \
  --num-workers 4 \
  --prior-alpha 0.035 \
  --seed 3201 \
  2>&1 | tee "$LOG"
