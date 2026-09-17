#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src
LOG="$PWD/outputs/bridge_logs/r255_seeta_preprocess.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "[R255-SEETA-PREPROCESS] start $(date -Is)"
export PATH="/root/miniconda3/bin:$PATH"
export nnUNet_raw="$PWD/outputs/nnunet/r202/nnUNet_raw"
export nnUNet_preprocessed="$PWD/outputs/nnunet/r202/nnUNet_preprocessed"
export nnUNet_results="$PWD/outputs/nnunet/r202/nnUNet_results"
/root/miniconda3/bin/python scripts/prepare_r255_nnunet_trainval_dataset.py --overwrite
nnUNetv2_plan_and_preprocess -d 202 -c 2d --verify_dataset_integrity
cp outputs/nnunet/r202/splits_final_r202.json \
  outputs/nnunet/r202/nnUNet_preprocessed/Dataset202_TSRS_RSNAEpiphysis2D/splits_final.json
echo "[R255-SEETA-PREPROCESS] done $(date -Is)"
