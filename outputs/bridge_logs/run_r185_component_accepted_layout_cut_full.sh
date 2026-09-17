#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

PY="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"

"${PY}" scripts/apply_r185_component_accepted_layout_cut.py \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --apply-anchor-exp r110_r100_r108_patch_basic \
  --output-exp r185_component_accepted_layout_cut \
  --val-summary-json outputs/analysis/r185_component_accepted_layout_cut_val_summary.json \
  --metrics-json outputs/analysis/r185_component_accepted_layout_cut_clean_test_v2_metrics.json \
  --result-json outputs/analysis/r185_component_accepted_layout_cut_result_summary.json \
  --peak-fracs 0.35,0.45,0.55,0.65 \
  --valley-fracs 0.12,0.16,0.22,0.28 \
  --min-areas 8,16,24,32 \
  --max-remove-fracs 0.0005,0.001,0.002,0.004 \
  --large-area-percentiles 40,50,60 \
  --min-val-edited-frac 0.000001 \
  --val-dice-drop-tol 0.0005 \
  --val-boundary-drop-tol 0.0 \
  --val-component-tol 0.0
