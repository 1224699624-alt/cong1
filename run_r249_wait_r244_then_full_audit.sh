#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
mkdir -p outputs/analysis outputs/bridge_logs

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
R244_PROC='scripts/audit_r244_gtfree_merge_targeted_candidates.py'
R244_CSV='outputs/analysis/r244_gtfree_merge_targeted_fullval_candidates.csv'
R244_JSON='outputs/analysis/r244_gtfree_merge_targeted_fullval_summary.json'

echo "[R249] waiting for R244 process to finish..."
while pgrep -f "${R244_PROC}" >/dev/null 2>&1; do
  date '+[R249] %Y-%m-%d %H:%M:%S still waiting for R244'
  if [ -f "${R244_JSON}" ]; then
    "${PY}" - <<'PY'
import json
p = "outputs/analysis/r244_gtfree_merge_targeted_fullval_summary.json"
try:
    d = json.load(open(p))
    print("[R249] R244 summary images=", d.get("num_candidate_images"), "rows=", d.get("num_candidate_rows"))
except Exception as exc:
    print("[R249] summary read failed:", type(exc).__name__, exc)
PY
  fi
  sleep 300
done

echo "[R249] R244 process finished; checking final files."
test -s "${R244_CSV}"
test -s "${R244_JSON}"

"${PY}" scripts/search_r245_gtfree_merge_filter.py \
  --input-csv "${R244_CSV}" \
  --output-json outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.json \
  --output-csv outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.csv \
  --min-selected 5 \
  --min-images 5 \
  --top-k-per-image 1

"${PY}" scripts/audit_r246_gtfree_merge_feature_failure.py \
  --candidate-csv "${R244_CSV}" \
  --r245-json outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.json \
  --output-json outputs/analysis/r246_gtfree_merge_feature_failure.fullval.json \
  --output-csv outputs/analysis/r246_gtfree_merge_feature_failure_feature_table.fullval.csv

echo "[R249] full R245/R246 audit complete."
