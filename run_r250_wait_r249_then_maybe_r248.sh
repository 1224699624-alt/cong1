#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
mkdir -p outputs/analysis outputs/bridge_logs

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
R249_PROC='run_r249_wait_r244_then_full_audit.sh'
R245_JSON='outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.json'
R248_JSON='outputs/analysis/r248_r244_context_crop_scorer_fullval.json'
R248_LOG='outputs/bridge_logs/r248_r244_context_crop_scorer_fullval.log'

echo "[R250] waiting for R249 full R245/R246 audit to finish..."
while pgrep -f "${R249_PROC}" >/dev/null 2>&1; do
  date '+[R250] %Y-%m-%d %H:%M:%S still waiting for R249'
  sleep 300
done

echo "[R250] R249 process finished; checking R245 decision."
if [ ! -s "${R245_JSON}" ]; then
  echo "[R250] Missing ${R245_JSON}; not launching R248."
  exit 2
fi

DECISION="$("${PY}" - <<'PY'
import json
p = "outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.json"
d = json.load(open(p))
print(d.get("decision", ""))
PY
)"

PASSING="$("${PY}" - <<'PY'
import json
p = "outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.json"
d = json.load(open(p))
print(int(d.get("num_passing_configs") or 0))
PY
)"

echo "[R250] R245 decision=${DECISION} passing_configs=${PASSING}"

if [ -s "${R248_JSON}" ]; then
  echo "[R250] R248 output already exists; not relaunching."
  exit 0
fi

if [ "${DECISION}" = "go_next_original_val_mask_editor" ] || [ "${PASSING}" -gt 0 ]; then
  echo "[R250] R245 passed a candidate gate; not launching R248 automatically."
  exit 0
fi

echo "[R250] R245 remains no-go; launching R248 full original-val grouped-CV."
./run_r248_r244_context_crop_scorer_fullval.sh 2>&1 | tee "${R248_LOG}"
