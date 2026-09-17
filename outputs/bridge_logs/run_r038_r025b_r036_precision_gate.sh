#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="TSRS_RSNA-Epiphysis"
CLEAN_DATASET="TSRS_RSNA-Epiphysis_clean_test_v2"
R036_ID="r036_medsam_gbc_adapter"
R036_NORM="r038_r036_medsam_gbc_adapter_norm"
R038_ID="r038_r025b_r036_precision_gate"
R036_CKPT="outputs/medsam_finetune/${DATASET}/${R036_ID}/best_mask_decoder.pt"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
MEDSAM_CHECKPOINT="${MEDSAM_CHECKPOINT:-checkpoints/medsam_vit_b.pth}"

mkdir -p outputs/analysis outputs/bridge_logs outputs/ablations_variants/${R036_NORM}

if [ ! -d "outputs/medsam_predictions/${R036_ID}/${DATASET}/train/masks" ]; then
  python scripts/infer_yolo_medsam.py \
    --dataset "${DATASET}" \
    --split train \
    --raw-root data/raw \
    --out-root "outputs/medsam_predictions/${R036_ID}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --medsam-checkpoint "${MEDSAM_CHECKPOINT}" \
    --finetuned-checkpoint "${R036_CKPT}" \
    --imgsz 1024 \
    --conf 0.20 \
    --iou 0.50 \
    --max-det 128 \
    --box-padding-ratio 0.05
fi

mkdir -p \
  "outputs/ablations_variants/${R036_NORM}/${DATASET}/train" \
  "outputs/ablations_variants/${R036_NORM}/${DATASET}/val" \
  "outputs/ablations_variants/${R036_NORM}/${CLEAN_DATASET}/test"
rm -f \
  "outputs/ablations_variants/${R036_NORM}/${DATASET}/train/masks" \
  "outputs/ablations_variants/${R036_NORM}/${DATASET}/val/masks" \
  "outputs/ablations_variants/${R036_NORM}/${CLEAN_DATASET}/test/masks"
ln -s "../../../../medsam_predictions/${R036_ID}/${DATASET}/train/masks" \
  "outputs/ablations_variants/${R036_NORM}/${DATASET}/train/masks"
ln -s "../../../../medsam_predictions/${R036_ID}/${DATASET}/val/masks" \
  "outputs/ablations_variants/${R036_NORM}/${DATASET}/val/masks"
ln -s "../../../../medsam_predictions/${R036_ID}_variants/${CLEAN_DATASET}/test/masks" \
  "outputs/ablations_variants/${R036_NORM}/${CLEAN_DATASET}/test/masks"

python scripts/train_anchor_pixel_residual.py \
  --train-dataset "${DATASET}" \
  --apply-dataset "${CLEAN_DATASET}" \
  --source-apply-dataset "${DATASET}" \
  --train-split train \
  --tune-split val \
  --apply-split test \
  --ablations-roots outputs/ablations_variants outputs/ablations \
  --anchor-exp r025b_anchor_local_pixel_residual_sam_hqsam \
  --candidate-exps "${R036_NORM}" \
  --output-exp "${R038_ID}" \
  --checkpoint "outputs/anchor_pixel_residual/${R038_ID}.pt" \
  --metrics-json "outputs/analysis/${R038_ID}_clean_test_v2_metrics.json" \
  --samples-per-image 3072 \
  --hidden 48 \
  --epochs 16 \
  --batch-size 65536 \
  --lr 0.002 \
  --boundary-radii 1,2,4,6,8,12 \
  --prob-thresholds 0.50,0.55,0.60,0.65,0.70,0.75,0.80 \
  --edit-margins 0.05,0.10,0.15,0.20,0.25,0.30 \
  --boundary-kernel 3 \
  --seed 20260638

cat "outputs/analysis/${R038_ID}_clean_test_v2_metrics.json"
