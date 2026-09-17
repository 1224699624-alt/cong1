#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"

"${PY}" scripts/train_r177_separation_arbitrator.py \
  --anchor-exp r100_like_r025b_r097_patch_basic_trainval \
  --apply-anchor-exp r110_r100_r108_patch_basic \
  --output-exp r177_boundary_preserving_separation_arbitrator \
  --checkpoint outputs/r177_separation_arbitrator/r177_boundary_preserving_separation_arbitrator.pt \
  --metrics-json outputs/analysis/r177_boundary_preserving_separation_arbitrator_clean_test_v2_metrics.json \
  --val-summary-json outputs/analysis/r177_boundary_preserving_separation_arbitrator_val_summary.json \
  --manifest-csv outputs/analysis/r177_bridge_risk_manifest/r177_bridge_risk_manifest.csv \
  --samples-per-risk-image 8192 \
  --samples-per-normal-image 1024 \
  --epochs 10 \
  --batch-size 131072 \
  --hidden 48 \
  --lr 0.001 \
  --pos-weight-scale 1.0 \
  --risk-radius 9 \
  --boundary-radius 3 \
  --zone-radii 3,5,7 \
  --prob-thresholds 0.45,0.50,0.55,0.60 \
  --edit-margins 0.05,0.10,0.15,0.20 \
  --seed 202607177

"${PY}" scripts/monitor_r177_boundary_preserving_result.py
