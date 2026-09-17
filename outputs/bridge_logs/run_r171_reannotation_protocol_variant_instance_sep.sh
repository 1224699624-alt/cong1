#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
VARIANT="TSRS_RSNA-Epiphysis_reannotation_protocol_reviewed_v1"
VARIANT_DIR="data/raw_variants/${VARIANT}"
VARIANT_META="${VARIANT_DIR}/r170_reannotation_protocol_metadata.json"
R170_STATUS="outputs/analysis/r170_reannotation_protocol_variant_build_status.json"

if [[ ! -d "${VARIANT_DIR}" ]]; then
  echo "R171 guard: missing R170 reviewed variant directory: ${VARIANT_DIR}" >&2
  exit 12
fi

if [[ ! -f "${VARIANT_META}" ]]; then
  echo "R171 guard: missing R170 metadata: ${VARIANT_META}" >&2
  exit 13
fi

if [[ ! -f "${R170_STATUS}" ]]; then
  echo "R171 guard: missing R170 build status: ${R170_STATUS}" >&2
  exit 14
fi

"${PY}" - <<'PY'
import json
from pathlib import Path

variant = Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotation_protocol_reviewed_v1")
meta = json.loads((variant / "r170_reannotation_protocol_metadata.json").read_text())
status = json.loads(Path("outputs/analysis/r170_reannotation_protocol_variant_build_status.json").read_text())
if meta.get("status") != "created":
    raise SystemExit(f"R171 guard: variant metadata status is not created: {meta.get('status')}")
if status.get("status") != "created":
    raise SystemExit(f"R171 guard: R170 build status is not created: {status.get('status')}")
policy = meta.get("policy", {})
if not policy.get("does_not_use_clean_test_v2_for_training"):
    raise SystemExit("R171 guard: metadata does not forbid clean-test-v2 training use.")
if not policy.get("does_not_use_reannotated_test"):
    raise SystemExit("R171 guard: metadata does not forbid reannotated test use.")
if policy.get("success_eval_remains") != "TSRS_RSNA-Epiphysis_clean_test_v2/test":
    raise SystemExit("R171 guard: success eval policy is not clean-test-v2/test.")
for split in ["train", "val", "test"]:
    images = {p.stem for p in (variant / split).glob("*") if p.is_file()}
    labels = {p.stem for p in (variant / f"{split}_labels").glob("*.png") if p.is_file()}
    if images != labels:
        raise SystemExit(f"R171 guard: unpaired {split}: images={len(images)} labels={len(labels)}")
orig_test = {p.stem for p in Path("data/raw/TSRS_RSNA-Epiphysis/test").glob("*") if p.is_file()}
variant_test = {p.stem for p in (variant / "test").glob("*") if p.is_file()}
if variant_test != orig_test:
    raise SystemExit("R171 guard: variant test must equal original test control.")
print(json.dumps({
    "r171_guard": "ok",
    "variant": str(variant),
    "status": meta.get("status"),
    "split_summary": meta.get("split_summary", {}),
}, indent=2))
PY

"${PY}" scripts/train_timm_instance_separation_segmenter.py \
  --dataset "${VARIANT}" \
  --raw-root data/raw_variants \
  --control-raw-root data/raw \
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
  --output-exp r171_reannotation_protocol_dinov3_instance_sep \
  --checkpoint outputs/timm_instance_sep/r171_reannotation_protocol_dinov3_instance_sep/best.pt \
  --history-json outputs/timm_instance_sep/r171_reannotation_protocol_dinov3_instance_sep/history.json \
  --metrics-json outputs/analysis/r171_reannotation_protocol_dinov3_instance_sep_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r171_reannotation_protocol_dinov3_instance_sep_original_test_metrics.json \
  --seed 20260771 \
  --device cuda

"${PY}" scripts/monitor_r171_reannotation_protocol_result.py \
  --metrics-json outputs/analysis/r171_reannotation_protocol_dinov3_instance_sep_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r171_reannotation_protocol_dinov3_instance_sep_original_test_metrics.json \
  --history-json outputs/timm_instance_sep/r171_reannotation_protocol_dinov3_instance_sep/history.json \
  --output-json outputs/analysis/r171_reannotation_protocol_dinov3_instance_sep_result_summary.json
