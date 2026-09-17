#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python
OUT_EXP=r110_r100_r108_patch_basic_trainval

for split in train val; do
  "${PY}" scripts/apply_patch_disagreement_arbitrator.py \
    --checkpoint outputs/patch_arbitrator/r110_r100_r108_patch_basic.pt \
    --dataset TSRS_RSNA-Epiphysis \
    --raw-root data/raw \
    --split "${split}" \
    --anchor-exp r100_like_r025b_r097_patch_basic_trainval \
    --candidate-exps r108_dinov3_convnext_tiny_unet \
    --output-exp "${OUT_EXP}" \
    --metrics-json "outputs/analysis/${OUT_EXP}_${split}_metrics.json" \
    --feature-mode basic \
    --zone-mode anchor_candidate_disagreement \
    --radius 0 \
    --threshold 0.50 \
    --margin 0.15
done

"${PY}" - <<'PY'
import json
from pathlib import Path

root = Path("outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis")
summary = {"output_exp": "r110_r100_r108_patch_basic_trainval", "splits": {}}
ok = True
for split, expected in [("train", 875), ("val", 96)]:
    mask_dir = root / split / "masks"
    count = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
    metrics_path = Path("outputs/analysis") / f"r110_r100_r108_patch_basic_trainval_{split}_metrics.json"
    summary["splits"][split] = {
        "mask_dir": str(mask_dir),
        "mask_count": count,
        "expected_count": expected,
        "metrics_json": str(metrics_path),
        "metrics_exists": metrics_path.exists(),
    }
    ok = ok and count == expected and metrics_path.exists()
summary["status"] = "ready" if ok else "incomplete"
out = Path("outputs/analysis/r183_r110_trainval_materialization_summary.json")
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
PY
