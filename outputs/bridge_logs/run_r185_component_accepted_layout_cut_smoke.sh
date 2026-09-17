#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

PY="/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"

"${PY}" scripts/apply_r185_component_accepted_layout_cut.py \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --apply-anchor-exp r110_r100_r108_patch_basic \
  --output-exp r185_component_accepted_layout_cut_smoke \
  --val-summary-json outputs/analysis/r185_component_accepted_layout_cut_smoke_val_summary.json \
  --metrics-json outputs/analysis/r185_component_accepted_layout_cut_smoke_clean_test_v2_metrics.json \
  --result-json outputs/analysis/r185_component_accepted_layout_cut_smoke_result_summary.json \
  --max-val-images 12 \
  --max-apply-images 0 \
  --peak-fracs 0.35,0.55 \
  --valley-fracs 0.16,0.28 \
  --min-areas 8,24 \
  --max-remove-fracs 0.001,0.004 \
  --large-area-percentiles 50 \
  --min-val-edited-frac 0.000001 \
  --val-dice-drop-tol 0.0005 \
  --val-boundary-drop-tol 0.0 \
  --val-component-tol 0.0
