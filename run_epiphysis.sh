#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/prepare_yolo_dataset.py --dataset TSRS_RSNA-Epiphysis --padding-ratio 0.08 --overwrite
python scripts/train_yolo.py \
  --data configs/TSRS_RSNA-Epiphysis.yaml \
  --model yolov8l.pt \
  --imgsz 1024 \
  --epochs 100 \
  --batch 4 \
  --name epiphysis_yolov8l_1024

python scripts/download_sam_checkpoint.py
python scripts/infer_yolo_sam.py \
  --dataset TSRS_RSNA-Epiphysis \
  --split test \
  --yolo-weights runs/detect/outputs/yolo/epiphysis_yolov8l_1024/weights/best.pt \
  --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --imgsz 1024 \
  --conf 0.20 \
  --box-padding-ratio 0.05 \
  --save-overlays

python scripts/evaluate_masks.py \
  --dataset TSRS_RSNA-Epiphysis \
  --split test \
  --pred-dir outputs/predictions/TSRS_RSNA-Epiphysis/test/masks
