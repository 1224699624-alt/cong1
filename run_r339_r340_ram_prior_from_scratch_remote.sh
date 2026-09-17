#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src
R339=outputs/ram_w600/r339_prior_from_scratch
mkdir -p "$R339" outputs/bridge_logs
/root/miniconda3/bin/python -u scripts/train_r339_ram_prior_from_scratch.py \
  --dataset-root data/remote_variants/RAM-W600 \
  --baseline-checkpoint outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth \
  --output "$R339" --epochs 60 --learning-rate 2e-4 \
  2>&1 | tee outputs/bridge_logs/r339_prior_from_scratch.log

PROMOTE=$(/root/miniconda3/bin/python -c "import json; print(int(json.load(open('$R339/result.json'))['promote_to_r340']))")
if [[ "$PROMOTE" != "1" ]]; then
  echo "R339 did not pass maturity/effectiveness gate; R340 not launched." | tee outputs/bridge_logs/r340_gate.log
  exit 0
fi

R340=outputs/ram_w600/r340_frozen_new_prior_backprop
mkdir -p "$R340"
/root/miniconda3/bin/python -u scripts/train_r338_ram_frozen_differentiable_prior.py \
  --dataset-root data/remote_variants/RAM-W600 \
  --baseline-checkpoint outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth \
  --prior-checkpoint "$R339/instance_prior_best.pth" \
  --r325-cache outputs/ram_w600/r332_joint_iterative_refinement/r325_official_cache.json \
  --r332-result outputs/ram_w600/r332_joint_iterative_refinement/steps_3/result.json \
  --output "$R340" --epochs 30 --warmup-epochs 4 \
  2>&1 | tee outputs/bridge_logs/r340_frozen_new_prior_backprop.log
