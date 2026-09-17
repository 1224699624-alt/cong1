#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src
export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

run_lambda() {
  local weight="$1"
  local tag="$2"
  /root/miniconda3/bin/python scripts/train_r324b_ram_nnunet_overlap_iem_lambda.py \
    --dataset-root data/remote_variants/RAM-W600 \
    --baseline-checkpoint outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth \
    --plain-result outputs/ram_w600/r324_nnunet_explicit_overlap_iem/result.json \
    --relation-graph outputs/ram_w600/r324_nnunet_explicit_overlap_iem/overlap_relation_graph.json \
    --output "outputs/ram_w600/r324_${tag}_nnunet_explicit_overlap_iem" \
    --prior-weight "$weight" --epochs 30 --min-epochs 15 --patience 10 \
    --learning-rate 5e-5 --size 384 --seed 3241
}

run_lambda 0.004 lambda004
run_lambda 0.006 lambda006
run_lambda 0.010 lambda010
