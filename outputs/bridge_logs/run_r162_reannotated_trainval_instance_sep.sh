#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
VARIANT="TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1"
VARIANT_DIR="data/raw_variants/${VARIANT}"
VARIANT_META="${VARIANT_DIR}/reannotated_trainval_only_metadata.json"

if [[ ! -d "${VARIANT_DIR}" ]]; then
  echo "R162 guard: missing train/val-only reannotated variant directory: ${VARIANT_DIR}" >&2
  exit 12
fi

if [[ ! -f "${VARIANT_META}" ]]; then
  echo "R162 guard: missing train/val-only metadata: ${VARIANT_META}" >&2
  exit 13
fi

"${PY}" - <<'PY'
import json
from pathlib import Path

variant = Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1")
meta = json.loads((variant / "reannotated_trainval_only_metadata.json").read_text())
policy = meta.get("policy", "")
if "reannotated test split is excluded" not in policy:
    raise SystemExit("R162 guard: metadata does not explicitly exclude reannotated test.")
if "clean-test-v2/test" not in policy and "TSRS_RSNA-Epiphysis_clean_test_v2/test" not in policy:
    raise SystemExit("R162 guard: metadata does not preserve clean-test-v2 success policy.")
expected = {
    "train": 1256,
    "val": 96,
    "test": 97,
}
for split, count in expected.items():
    image_count = len([p for p in (variant / split).glob("*") if p.is_file()])
    label_count = len([p for p in (variant / f"{split}_labels").glob("*.png") if p.is_file()])
    if image_count != count or label_count != count:
        raise SystemExit(f"R162 guard: unexpected {split} counts images={image_count} labels={label_count} expected={count}")
orig_test = {p.stem for p in Path("data/raw/TSRS_RSNA-Epiphysis/test").glob("*") if p.is_file()}
variant_test = {p.stem for p in (variant / "test").glob("*") if p.is_file()}
re_test = {p.stem for p in Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1/test").glob("*") if p.is_file()}
if variant_test != orig_test:
    raise SystemExit("R162 guard: variant test is not exactly original test.")
if variant_test == re_test:
    raise SystemExit("R162 guard: variant test unexpectedly equals reannotated test.")
print(json.dumps({"r162_guard": "ok", "variant": str(variant), "splits": expected}, indent=2))
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
  --output-exp r162_reannotated_trainval_dinov3_instance_sep \
  --checkpoint outputs/timm_instance_sep/r162_reannotated_trainval_dinov3_instance_sep/best.pt \
  --history-json outputs/timm_instance_sep/r162_reannotated_trainval_dinov3_instance_sep/history.json \
  --metrics-json outputs/analysis/r162_reannotated_trainval_dinov3_instance_sep_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r162_reannotated_trainval_dinov3_instance_sep_original_test_metrics.json \
  --seed 20260762 \
  --device cuda
