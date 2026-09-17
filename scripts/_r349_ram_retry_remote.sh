#!/bin/bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
while pgrep -f "nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR349ParameterPCGrad" >/dev/null; do sleep 60; done
cd "$ROOT"
export PATH=/root/miniconda3/bin:$PATH PYTHONPATH=$ROOT/scripts CUDA_VISIBLE_DEVICES=0
OUT=outputs/ram_w600/r349_parameter_pcgrad_retry
rm -rf "$OUT"
mkdir -p "$OUT"
python scripts/train_r349_ram_parameter_pcgrad.py --dataset-root data/remote_variants/RAM-W600 --prior-root outputs/priors/r323_ram_native_r317_single_seed --r332-checkpoint outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth --output "$OUT/audit" --audit 2>&1 | tee outputs/bridge_logs/r349_ram_retry_audit.log
python -c "import json; assert json.load(open('$OUT/audit/smoke_audit.json'))['passed']"
python scripts/train_r349_ram_parameter_pcgrad.py --dataset-root data/remote_variants/RAM-W600 --prior-root outputs/priors/r323_ram_native_r317_single_seed --r332-checkpoint outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth --output "$OUT/full" --epochs 30 2>&1 | tee outputs/bridge_logs/r349_ram_retry_full.log
