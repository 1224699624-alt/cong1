#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
EXTERNAL="/root/autodl-tmp/external"
PYTHON="/root/miniconda3/bin/python"
cd "$ROOT"

mkdir -p outputs/bridge_logs outputs/smoke outputs/experiments/r320_multibackbone outputs/analysis

report_disk() {
  echo "[R320 disk] $(date -Is)"
  df -h "$ROOT" "$EXTERNAL" | awk 'NR == 1 || !seen[$1]++'
  du -sh outputs/experiments/r320_multibackbone outputs/cache/r320_medsam_embeddings "$EXTERNAL/checkpoints" 2>/dev/null || true
}

require_free_gib() {
  local minimum_gib="$1"
  local stage="$2"
  local free_kib
  free_kib=$(df -Pk "$ROOT" | awk 'NR == 2 {print $4}')
  report_disk
  if (( free_kib < minimum_gib * 1024 * 1024 )); then
    echo "[R320 disk guard] refusing to start ${stage}: less than ${minimum_gib} GiB free" >&2
    exit 75
  fi
}

wait_for_result() {
  local result="$1"
  local screen_pattern="$2"
  while [[ ! -f "$result" ]]; do
    if ! screen -ls 2>/dev/null | grep -q "$screen_pattern"; then
      echo "required run exited without result: $result" >&2
      exit 1
    fi
    sleep 30
  done
}

require_file_size() {
  local path="$1"
  local expected="$2"
  local current=0
  [[ -f "$path" ]] && current=$(stat -c %s "$path")
  if [[ "$current" -ne "$expected" ]]; then
    echo "[R320 asset gate] missing or incomplete: $path $current/$expected" >&2
    echo "run run_r320_prepare_assets.sh first; training was not started" >&2
    exit 66
  fi
}

evaluate_pair() {
  local model_root="$1"
  local prefix="$2"
  local variant
  for variant in plain prior; do
    local output="outputs/analysis/${prefix}_${variant}_original_val_r201.json"
    if "$PYTHON" - "$output" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    result = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if result.get("num_images") == 96 and isinstance(result.get("mean"), dict) else 1)
PY
    then
      echo "[R320 queue] reuse valid R201 result: $output"
      continue
    fi
    "$PYTHON" scripts/evaluate_r201_single_split.py \
      --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
      --pred-dir "$model_root/$variant/masks" \
      --output "$output" \
      --expected-count 96
  done
}

echo "[R320 queue] waiting for exact U-Net pair"
wait_for_result \
  outputs/experiments/r320_multibackbone/unet_seed3201_exact/result.json \
  r320_unet_full_exact
evaluate_pair outputs/experiments/r320_multibackbone/unet_seed3201_exact r320_unet_seed3201

echo "[R320 queue] waiting for TransUNet weights"
require_file_size "$EXTERNAL/checkpoints/R50_ViT-B_16.npz" 461217452
require_free_gib 8 "TransUNet smoke/full pair"

echo "[R320 queue] TransUNet smoke"
"$PYTHON" scripts/run_r320_paired_external_backbone.py \
  --backbone transunet \
  --external-root "$EXTERNAL" \
  --transunet-npz "$EXTERNAL/checkpoints/R50_ViT-B_16.npz" \
  --output-root outputs/smoke/r320_transunet_exact \
  --img-size 512 --batch-size 1 --num-workers 2 \
  --epochs 2 --min-epochs 1 --patience 2 \
  --limit-train 8 --limit-val 4 --seed 3201 --prior-alpha 0.035 \
  2>&1 | tee outputs/bridge_logs/r320_transunet_smoke.log

echo "[R320 queue] TransUNet mature pair"
"$PYTHON" scripts/run_r320_paired_external_backbone.py \
  --backbone transunet \
  --external-root "$EXTERNAL" \
  --transunet-npz "$EXTERNAL/checkpoints/R50_ViT-B_16.npz" \
  --output-root outputs/experiments/r320_multibackbone/transunet_seed3201 \
  --img-size 512 --batch-size 1 --num-workers 4 \
  --epochs 60 --min-epochs 20 --patience 12 \
  --seed 3201 --prior-alpha 0.035 \
  2>&1 | tee outputs/bridge_logs/r320_transunet_full.log
evaluate_pair outputs/experiments/r320_multibackbone/transunet_seed3201 r320_transunet_seed3201
require_free_gib 8 "Swin-UMamba smoke/full pair"

echo "[R320 queue] waiting for Swin-UMamba weights and dependencies"
require_file_size "$EXTERNAL/checkpoints/vmamba_tiny_e292.pth" 91649482
"$PYTHON" -c 'import monai, causal_conv1d, mamba_ssm' >/dev/null

echo "[R320 queue] Swin-UMamba smoke"
"$PYTHON" scripts/run_r320_paired_external_backbone.py \
  --backbone swin_umamba \
  --external-root "$EXTERNAL" \
  --swin-pretrained "$EXTERNAL/checkpoints/vmamba_tiny_e292.pth" \
  --output-root outputs/smoke/r320_swin_umamba_exact \
  --img-size 512 --batch-size 1 --num-workers 2 \
  --epochs 2 --min-epochs 1 --patience 2 \
  --limit-train 8 --limit-val 4 --seed 3201 --prior-alpha 0.035 \
  2>&1 | tee outputs/bridge_logs/r320_swin_umamba_smoke.log

echo "[R320 queue] Swin-UMamba mature pair"
"$PYTHON" scripts/run_r320_paired_external_backbone.py \
  --backbone swin_umamba \
  --external-root "$EXTERNAL" \
  --swin-pretrained "$EXTERNAL/checkpoints/vmamba_tiny_e292.pth" \
  --output-root outputs/experiments/r320_multibackbone/swin_umamba_seed3201 \
  --img-size 512 --batch-size 1 --num-workers 4 \
  --epochs 60 --min-epochs 20 --patience 12 \
  --seed 3201 --prior-alpha 0.035 \
  2>&1 | tee outputs/bridge_logs/r320_swin_umamba_full.log
evaluate_pair outputs/experiments/r320_multibackbone/swin_umamba_seed3201 r320_swin_umamba_seed3201

echo "[R320 queue] waiting for MedSAM checkpoint"
require_file_size "$EXTERNAL/checkpoints/medsam_vit_b.pth" 375049145
require_free_gib 10 "prompt-free MedSAM cache and paired run"

echo "[R320 queue] prompt-free MedSAM smoke"
"$PYTHON" scripts/run_r320_paired_promptfree_medsam.py \
  --checkpoint "$EXTERNAL/checkpoints/medsam_vit_b.pth" \
  --medsam-repo "$EXTERNAL/MedSAM" \
  --cache-root outputs/cache/r320_medsam_embeddings \
  --output-root outputs/smoke/r320_promptfree_medsam_exact \
  --epochs 2 --min-epochs 1 --patience 2 --num-workers 2 \
  --limit-train 8 --limit-val 4 --seed 3201 --prior-alpha 0.035 \
  2>&1 | tee outputs/bridge_logs/r320_promptfree_medsam_smoke.log

echo "[R320 queue] prompt-free MedSAM mature pair"
"$PYTHON" scripts/run_r320_paired_promptfree_medsam.py \
  --checkpoint "$EXTERNAL/checkpoints/medsam_vit_b.pth" \
  --medsam-repo "$EXTERNAL/MedSAM" \
  --cache-root outputs/cache/r320_medsam_embeddings \
  --output-root outputs/experiments/r320_multibackbone/promptfree_medsam_seed3201 \
  --epochs 40 --min-epochs 15 --patience 10 --num-workers 2 \
  --seed 3201 --prior-alpha 0.035 \
  2>&1 | tee outputs/bridge_logs/r320_promptfree_medsam_full.log
evaluate_pair outputs/experiments/r320_multibackbone/promptfree_medsam_seed3201 r320_promptfree_medsam_seed3201
report_disk

echo "[R320 queue] all scheduled pairs complete"
