#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
cd "$ROOT"
export PATH=/root/miniconda3/bin:$PATH PYTHONPATH="$ROOT/scripts" CUDA_VISIBLE_DEVICES=0 nnUNet_compile=false nnUNet_n_proc_DA=0

RAM_OUT=outputs/ram_w600/r349_parameter_pcgrad
TSRS_RUN=outputs/bridge_logs/r349_tsrs_parameter_pcgrad
DATA=outputs/nnunet/r317_image_centers_only/data
export nnUNet_raw="$ROOT/$DATA/nnUNet_raw" nnUNet_preprocessed="$ROOT/$DATA/nnUNet_preprocessed" nnUNet_results="$ROOT/$TSRS_RUN/nnUNet_results"
export R349_TSRS_INIT_CHECKPOINT="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_best.pth"

rm -rf "$RAM_OUT" "$TSRS_RUN"
mkdir -p "$RAM_OUT" "$TSRS_RUN"

python scripts/smoke_r349_parameter_gradient_gate.py | tee outputs/bridge_logs/r349_gradient_unit_smoke.log
python scripts/train_r349_ram_parameter_pcgrad.py \
  --dataset-root data/remote_variants/RAM-W600 \
  --prior-root outputs/priors/r323_ram_native_r317_single_seed \
  --r332-checkpoint outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth \
  --output "$RAM_OUT/audit" --audit | tee outputs/bridge_logs/r349_ram_audit.log

TRAINER_DIR="$(python -c 'from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / "training/nnUNetTrainer/variants/loss")')"
cp scripts/nnunet_trainers/nnUNetTrainerR349ParameterPCGrad.py "$TRAINER_DIR/"
export R349_AUDIT_ONLY=true
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR349ParameterPCGrad | tee "$TSRS_RUN/audit.log"
test "$(python -c 'import json; print(str(json.load(open("outputs/ram_w600/r349_parameter_pcgrad/audit/smoke_audit.json"))["passed"]).lower())')" = true
TSRS_AUDIT="$(find "$TSRS_RUN/nnUNet_results" -name smoke_audit.json -print -quit)"
test -n "$TSRS_AUDIT"
test "$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' "$TSRS_AUDIT")" = true

unset R349_AUDIT_ONLY
rm -rf "$TSRS_RUN/nnUNet_results"
screen -dmS r349_ram bash -lc "cd '$ROOT' && export PATH=/root/miniconda3/bin:\$PATH PYTHONPATH='$ROOT/scripts' CUDA_VISIBLE_DEVICES=0 && python scripts/train_r349_ram_parameter_pcgrad.py --dataset-root data/remote_variants/RAM-W600 --prior-root outputs/priors/r323_ram_native_r317_single_seed --r332-checkpoint outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth --output '$RAM_OUT/full' --epochs 30 2>&1 | tee outputs/bridge_logs/r349_ram_full.log"
screen -dmS r349_tsrs_queue bash -lc "while screen -list | grep -q '[.]r349_ram'; do sleep 60; done; cd '$ROOT'; export PATH=/root/miniconda3/bin:\$PATH PYTHONPATH='$ROOT/scripts' CUDA_VISIBLE_DEVICES=0 nnUNet_compile=false nnUNet_n_proc_DA=0 nnUNet_raw='$ROOT/$DATA/nnUNet_raw' nnUNet_preprocessed='$ROOT/$DATA/nnUNet_preprocessed' nnUNet_results='$ROOT/$TSRS_RUN/nnUNet_results' R349_TSRS_INIT_CHECKPOINT='$R349_TSRS_INIT_CHECKPOINT'; nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR349ParameterPCGrad 2>&1 | tee '$TSRS_RUN/train.log'"
screen -ls
