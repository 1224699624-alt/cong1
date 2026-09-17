#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
DATA="$ROOT/data/remote_variants/RAM-W600"
BASE="$ROOT/outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth"
OUT="$ROOT/outputs/ram_w600/r326_layer_projection_prior_batched"
LOG="$ROOT/outputs/bridge_logs/r326_layer_projection_prior_batched.log"

cd "$ROOT"
mkdir -p "$OUT" outputs/bridge_logs
exec > >(tee -a "$LOG") 2>&1
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3261 PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"

echo "[R326] start $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
df -h /root/autodl-tmp
test "$(find "$DATA/BoneSegmentation/masks/train" -maxdepth 1 -name '*.npy' | wc -l)" -eq 425
test "$(find "$DATA/BoneSegmentation/masks/val" -maxdepth 1 -name '*.npy' | wc -l)" -eq 69
test -s "$BASE"
$PY -m py_compile scripts/train_r326_ram_nnunet_layer_projection_prior.py

echo "[R326] gradient smoke $(date -Is)"
$PY scripts/train_r326_ram_nnunet_layer_projection_prior.py \
  --dataset-root "$DATA" --baseline-checkpoint "$BASE" --output "$OUT/smoke" \
  --projection-weight 0.00025 --limit-train 4 --limit-val 2 --smoke

echo "[R326] paired adapters $(date -Is)"
$PY scripts/train_r326_ram_nnunet_layer_projection_prior.py \
  --dataset-root "$DATA" --baseline-checkpoint "$BASE" --output "$OUT" \
  --epochs 30 --min-epochs 15 --patience 10 --learning-rate 2e-4 \
  --projection-weight 0.00025 --size 384 --batch-size 4 --workers 4 --seed 3261

test -s "$OUT/result.json"
echo "[R326] complete $(date -Is)"
du -sh "$OUT"
df -h /root/autodl-tmp
