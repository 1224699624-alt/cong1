#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

PY="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"

"${PY}" scripts/apply_r186_fast_neck_gate.py \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --apply-anchor-exp r110_r100_r108_patch_basic \
  --output-exp r186_fast_neck_gate_smoke \
  --val-summary-json outputs/analysis/r186_fast_neck_gate_smoke_val_summary.json \
  --metrics-json outputs/analysis/r186_fast_neck_gate_smoke_clean_test_v2_metrics.json \
  --result-json outputs/analysis/r186_fast_neck_gate_smoke_result_summary.json \
  --max-val-images 8 \
  --max-apply-images 0 \
  --neck-widths 1,2 \
  --band-radii 2,5 \
  --min-areas 1,4 \
  --max-remove-fracs 0.0005,0.001 \
  --min-val-edited-frac 0.0000001 \
  --val-dice-drop-tol 0.00025 \
  --val-boundary-drop-tol 0.0 \
  --val-component-tol 0.0
