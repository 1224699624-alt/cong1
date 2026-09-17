#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
cd "$ROOT"
while screen -list | grep -q '[.]r346_tsrs'; do sleep 60; done
export PATH=/root/miniconda3/bin:$PATH PYTHONPATH="$ROOT/scripts" CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3461
RUN="$ROOT/outputs/bridge_logs/r346_ram_conflict_aware"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/train.log") 2>&1
/root/miniconda3/bin/python scripts/train_r335_ram_conditional_state_loss.py \
  --dataset-root data/remote_variants/RAM-W600 \
  --prior-root outputs/priors/r323_ram_native_r317_single_seed \
  --r332-checkpoint outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth \
  --output outputs/ram_w600/r346_conflict_aware_loss --epochs 60 --steps 2 \
  --seg-weight .10 --seam-weight .025 --overlap-weight .04 --preserve-weight 1.0 \
  --loss-version r346
