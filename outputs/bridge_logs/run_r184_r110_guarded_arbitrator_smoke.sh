#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"

"${PY}" scripts/prepare_r177_bridge_risk_manifest.py \
  --dataset TSRS_RSNA-Epiphysis \
  --raw-root data/raw \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --output-dir outputs/analysis/r184_r110_bridge_risk_manifest \
  --splits train,val \
  --gap-radius 7 \
  --min-risk-pixels 96 \
  --min-train-risk-images 5 \
  --min-val-risk-images 2 \
  --max-images-per-split 96

"${PY}" scripts/train_r184_r110_guarded_arbitrator.py \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --apply-anchor-exp r110_r100_r108_patch_basic \
  --output-exp r184_r110_guarded_arbitrator_smoke \
  --checkpoint outputs/r184_guarded_arbitrator/r184_r110_guarded_arbitrator_smoke.pt \
  --metrics-json outputs/analysis/r184_r110_guarded_arbitrator_smoke_clean_test_v2_metrics.json \
  --val-summary-json outputs/analysis/r184_r110_guarded_arbitrator_smoke_val_summary.json \
  --result-json outputs/analysis/r184_r110_guarded_arbitrator_smoke_result_summary.json \
  --manifest-csv outputs/analysis/r184_r110_bridge_risk_manifest/r177_bridge_risk_manifest.csv \
  --max-train-images 64 \
  --max-val-images 24 \
  --max-apply-images 0 \
  --samples-per-risk-image 2048 \
  --samples-per-normal-image 256 \
  --epochs 3 \
  --hidden 24 \
  --lr 0.0007 \
  --risk-radius 7 \
  --boundary-radius 2 \
  --zone-radii 2,3 \
  --prob-thresholds 0.55,0.60 \
  --edit-margins 0.25,0.30,0.35 \
  --max-edited-frac 0.004 \
  --seed 202607184
