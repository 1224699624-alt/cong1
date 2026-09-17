#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
WORK=/root/r351_workspace
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
mkdir -p "$WORK/logs"
MODE="${1:-audit}"
ARGS=(--dataset-root data/remote_variants/RAM-W600
 --prior-root outputs/priors/r323_ram_native_r317_single_seed
 --r332-checkpoint outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth
 --overlap-checkpoint "$WORK/overlap/full/best.pth"
 --native-checkpoint outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth
 --output "$WORK/ram_$MODE" --epochs 2 --aux-ramp-epochs 5)
if [[ "$MODE" == audit* ]]; then ARGS+=(--audit); fi
/root/miniconda3/bin/python scripts/train_r351_ram_dual_prior.py "${ARGS[@]}" 2>&1 | tee "$WORK/logs/ram_${MODE}.log"
