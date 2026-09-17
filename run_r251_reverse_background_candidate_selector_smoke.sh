#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
mkdir -p outputs/analysis outputs/models outputs/bridge_logs
stamp="$(date +%Y%m%d_%H%M%S)"
log="outputs/bridge_logs/r251_reverse_background_candidate_selector_smoke_${stamp}.log"
set +e
/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python scripts/audit_r251_reverse_background_candidate_selector.py \
  --dataset TSRS_RSNA-Epiphysis \
  --raw-root data/raw \
  --ablations-root outputs/ablations_variants \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --train-split train \
  --split val \
  --limit-train 2 \
  --max-scan-images 16 \
  --limit 2 \
  --dist-percentiles 6,8,10,12 \
  --train-max-candidates-per-percentile 16 \
  --max-candidates-generated 16 \
  --flush-every 1 \
  --output-json outputs/analysis/r251_reverse_background_candidate_selector_smoke.json \
  --candidate-csv outputs/analysis/r251_reverse_background_candidate_selector_smoke_candidates.csv \
  --grid-csv outputs/analysis/r251_reverse_background_candidate_selector_smoke_grid.csv \
  --progress-json outputs/analysis/r251_reverse_background_candidate_selector_smoke_progress.json \
  --model-path outputs/models/r251_reverse_background_candidate_selector_smoke.pkl \
  2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
cp "$log" outputs/bridge_logs/r251_reverse_background_candidate_selector_smoke.log
exit "$status"
