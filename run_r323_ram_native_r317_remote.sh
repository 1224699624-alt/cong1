#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
DATA="$ROOT/data/remote_variants/RAM-W600"
BASE="$ROOT/outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth"
PRIOR_DIR="$ROOT/outputs/pair_prior/r323_ram_native_r317_single_seed"
PRIOR_JSON="$ROOT/outputs/analysis/r323_ram_native_r317_single_seed.json"
MAPS="$ROOT/outputs/priors/r323_ram_native_r317_single_seed"
RUN="$ROOT/outputs/ram_w600/r323_nnunet_ram_native_r317"
LOG="$ROOT/outputs/bridge_logs/r323_ram_native_r317_long.log"

cd "$ROOT"
mkdir -p outputs/bridge_logs outputs/analysis "$PRIOR_DIR"
exec > >(tee -a "$LOG") 2>&1
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3231

echo "[R323] start $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
df -h /root/autodl-tmp
test "$(find "$DATA/BoneSegmentation/masks/train" -maxdepth 1 -name '*.npy' | wc -l)" -eq 425
test "$(find "$DATA/BoneSegmentation/masks/val" -maxdepth 1 -name '*.npy' | wc -l)" -eq 69
test -s "$BASE"
$PY -m py_compile scripts/train_r323_ram_r317_prior.py scripts/build_r323_ram_native_prior_maps.py scripts/train_r322_ram_nnunet_r317_seam_prior.py

echo "[R323] train RAM-native R317 prior: one seed, 60 full epochs $(date -Is)"
$PY scripts/train_r323_ram_r317_prior.py \
  --dataset-root "$DATA" --baseline-checkpoint "$BASE" \
  --output-dir "$PRIOR_DIR" --result-json "$PRIOR_JSON" \
  --epochs 60 --seed 3231 --batch-size 16 --workers 4 --lr 3e-4

SELECTED="$PRIOR_DIR/ram_image_centers_seed3231_selected.pt"
test -s "$SELECTED"
echo "[R323] build frozen GT-free train/val maps $(date -Is)"
$PY scripts/build_r323_ram_native_prior_maps.py \
  --dataset-root "$DATA" --baseline-checkpoint "$BASE" \
  --prior-checkpoint "$SELECTED" --output-root "$MAPS" --device cuda
test "$(find "$MAPS/train" -maxdepth 1 -name '*.npy' | wc -l)" -eq 425
test "$(find "$MAPS/val" -maxdepth 1 -name '*.npy' | wc -l)" -eq 69

echo "[R323] paired nnU-Net validation: max 30, minimum 15 epochs $(date -Is)"
$PY scripts/train_r322_ram_nnunet_r317_seam_prior.py \
  --dataset-root "$DATA" --prior-root "$MAPS" --baseline-checkpoint "$BASE" \
  --output "$RUN" --epochs 30 --min-epochs 15 --patience 10 \
  --alpha .035 --size 384 --seed 3231 \
  --experiment-id R323_RAM_NNUNET_RAM_NATIVE_R317

test -s "$RUN/result.json"
echo "[R323] complete $(date -Is)"
df -h /root/autodl-tmp
