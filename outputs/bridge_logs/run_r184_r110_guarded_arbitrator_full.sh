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
  --min-train-risk-images 20 \
  --min-val-risk-images 5

"${PY}" scripts/train_r184_r110_guarded_arbitrator.py \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --apply-anchor-exp r110_r100_r108_patch_basic \
  --output-exp r184_r110_guarded_arbitrator \
  --checkpoint outputs/r184_guarded_arbitrator/r184_r110_guarded_arbitrator.pt \
  --metrics-json outputs/analysis/r184_r110_guarded_arbitrator_clean_test_v2_metrics.json \
  --val-summary-json outputs/analysis/r184_r110_guarded_arbitrator_val_summary.json \
  --result-json outputs/analysis/r184_r110_guarded_arbitrator_result_summary.json \
  --manifest-csv outputs/analysis/r184_r110_bridge_risk_manifest/r177_bridge_risk_manifest.csv \
  --samples-per-risk-image 4096 \
  --samples-per-normal-image 512 \
  --epochs 6 \
  --hidden 32 \
  --lr 0.0007 \
  --risk-radius 7 \
  --boundary-radius 2 \
  --zone-radii 2,3,5 \
  --prob-thresholds 0.50,0.55,0.60 \
  --edit-margins 0.20,0.25,0.30,0.35 \
  --max-edited-frac 0.006 \
  --seed 202607184
