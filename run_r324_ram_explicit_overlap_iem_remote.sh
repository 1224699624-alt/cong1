#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
/root/miniconda3/bin/python scripts/train_r324_ram_nnunet_explicit_overlap_iem.py \
  --dataset-root data/remote_variants/RAM-W600 \
  --baseline-checkpoint outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth \
  --output outputs/ram_w600/r324_nnunet_explicit_overlap_iem \
  --epochs 30 --min-epochs 15 --patience 10 \
  --prior-weight 0.001 --learning-rate 5e-5 --size 384 \
  --seed 3241 --min-overlap-cases 3 --max-pairs 24
