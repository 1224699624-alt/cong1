#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
VARIANT="TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1"
VARIANT_DIR="data/raw_variants/${VARIANT}"
VARIANT_META="${VARIANT_DIR}/label_protocol_variant_metadata.json"
GATE_JSON="outputs/analysis/r134_label_protocol_review_manifest/r134_review_gate_status.json"

if [[ ! -d "${VARIANT_DIR}" ]]; then
  echo "R140 guard: missing reviewed variant directory: ${VARIANT_DIR}" >&2
  exit 12
fi

if [[ ! -f "${VARIANT_META}" ]]; then
  echo "R140 guard: missing reviewed variant metadata: ${VARIANT_META}" >&2
  exit 13
fi

"${PY}" scripts/check_label_protocol_review_manifest.py \
  --manifest outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.json \
  --csv outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.csv \
  --output-json "${GATE_JSON}"

"${PY}" - <<'PY'
import json
from pathlib import Path
gate = json.loads(Path("outputs/analysis/r134_label_protocol_review_manifest/r134_review_gate_status.json").read_text())
if not gate.get("gate_pass"):
    raise SystemExit("R140 guard: R134 gate is not passing; do not train on reviewed variant yet.")
meta = json.loads(Path("data/raw_variants/TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1/label_protocol_variant_metadata.json").read_text())
if meta.get("clean_test_policy") != "clean-test-v2 rows were diagnostic-only and were not used to build this variant":
    raise SystemExit("R140 guard: reviewed variant metadata lacks clean-test diagnostic-only policy.")
print(json.dumps({"r140_guard": "ok", "gate": gate["counts"], "variant_status": meta.get("status")}, indent=2))
PY

"${PY}" scripts/train_timm_instance_separation_segmenter.py \
  --dataset "${VARIANT}" \
  --raw-root data/raw_variants \
  --control-dataset TSRS_RSNA-Epiphysis \
  --eval-raw-root data/raw_variants \
  --eval-dataset TSRS_RSNA-Epiphysis_clean_test_v2 \
  --eval-split test \
  --encoder convnext_tiny.dinov3_lvd1689m \
  --pretrained \
  --img-size 512 \
  --epochs 36 \
  --min-epochs 10 \
  --patience 8 \
  --batch-size 4 \
  --num-workers 4 \
  --decoder-channels 128 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --boundary-loss-weight 0.18 \
  --sep-loss-weight 0.75 \
  --hover-loss-weight 0.20 \
  --bridge-gap-loss-weight 0.35 \
  --background-loss-weight 0.05 \
  --sep-band-kernel 13 \
  --sep-suppress-weights 0.30,0.45,0.60,0.75 \
  --thresholds 0.55,0.60,0.65,0.70,0.75,0.80 \
  --output-exp r140_reviewed_variant_dinov3_instance_sep \
  --checkpoint outputs/timm_instance_sep/r140_reviewed_variant_dinov3_instance_sep/best.pt \
  --history-json outputs/timm_instance_sep/r140_reviewed_variant_dinov3_instance_sep/history.json \
  --metrics-json outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_original_test_metrics.json \
  --seed 20260740 \
  --device cuda

"${PY}" scripts/monitor_r140_reviewed_variant_result.py \
  --metrics-json outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_original_test_metrics.json \
  --history-json outputs/timm_instance_sep/r140_reviewed_variant_dinov3_instance_sep/history.json \
  --output-json outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_result_summary.json
