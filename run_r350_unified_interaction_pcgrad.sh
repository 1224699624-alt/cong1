#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
PY=/root/miniconda3/bin
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
R317="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/$DATASET/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
R332="$ROOT/outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth"
TSRS="$ROOT/outputs/nnunet/r350_unified_interaction_pcgrad"
RAM="$ROOT/outputs/ram_w600/r350_unified_interaction_pcgrad"

cd "$ROOT"
mkdir -p outputs/bridge_logs outputs/analysis "$TSRS" "$RAM"
exec > >(tee -a outputs/bridge_logs/r350_unified_interaction_pcgrad.log) 2>&1
export PATH="$PY:$PATH" PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0
export nnUNet_compile=false nnUNet_n_proc_DA=0

test -s "$R317"
test -s "$R332"
test -d data/remote_variants/RAM-W600
test -d outputs/priors/r323_ram_native_r317_single_seed
test -d outputs/nnunet/r317_image_centers_only/imagesValPrior
test "$(find data/raw/TSRS_RSNA-Epiphysis/val_labels -maxdepth 1 -type f | wc -l)" -eq 96
! printf '%s\n' "$TSRS" "$RAM" | grep -qi 'clean-test\|articular'
avail_kb="$(df --output=avail /root/autodl-tmp | tail -1)"
test "$avail_kb" -ge 5242880

echo '[R350] shared CPU smoke'
$PY/python scripts/smoke_r350_unified_interaction_loss.py

echo '[R350] RAM real-data audit'
rm -rf "$RAM/audit"
$PY/python scripts/train_r350_ram_unified_interaction_pcgrad.py \
  --dataset-root data/remote_variants/RAM-W600 \
  --prior-root outputs/priors/r323_ram_native_r317_single_seed \
  --r332-checkpoint "$R332" --output "$RAM/audit" --audit
$PY/python -c "import json; assert json.load(open('$RAM/audit/smoke_audit.json'))['passed']"

TRAINER=nnUNetTrainerR350UnifiedInteractionPCGrad
TRAINER_DIR="$($PY/python -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/${TRAINER}.py "$TRAINER_DIR/${TRAINER}.py"
$PY/python -m py_compile scripts/r350_unified_interaction_loss.py scripts/train_r350_ram_unified_interaction_pcgrad.py "$TRAINER_DIR/${TRAINER}.py"

echo '[R350] TSRS real-data audit'
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$TSRS/audit_results" R350_TSRS_INIT_CHECKPOINT="$R317" R350_AUDIT_ONLY=true
rm -rf "$nnUNet_results"
nnUNetv2_train 204 2d 0 -tr "$TRAINER"
AUDIT="$(find "$nnUNet_results" -name smoke_audit.json -print -quit)"
test -n "$AUDIT"
$PY/python -c "import json,sys; assert json.load(open(sys.argv[1]))['passed']" "$AUDIT"

echo '[R350] TSRS full 30 epochs'
unset R350_AUDIT_ONLY
export nnUNet_results="$TSRS/nnUNet_results"
rm -rf "$nnUNet_results"
nnUNetv2_train 204 2d 0 -tr "$TRAINER"
FOLD="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__2d/fold_0"
test -s "$FOLD/checkpoint_best.pth"
test -s "$FOLD/checkpoint_final.pth"
test "$(wc -l < "$FOLD/r350_dynamics.jsonl")" -eq 30

evaluate_tsrs () {
  local checkpoint="$1" tag="$2" pred="$TSRS/predictions_$2" masks="$TSRS/masks_$2"
  rm -rf "$pred" "$masks"
  nnUNetv2_predict -i outputs/nnunet/r317_image_centers_only/imagesValPrior -o "$pred" \
    -d 204 -c 2d -f 0 -tr "$TRAINER" -chk "$checkpoint"
  $PY/python scripts/convert_nnunet_predictions.py --pred-dir "$pred" --mask-dir "$masks" \
    --expected-count 96 --strip-prefix val_
  $PY/python scripts/evaluate_r201_single_split.py \
    --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$masks" \
    --output "outputs/analysis/r350_tsrs_unified_${tag}_original_val_r201.json" --expected-count 96
  rm -rf "$pred"
}
evaluate_tsrs checkpoint_best.pth best
evaluate_tsrs checkpoint_final.pth final
rm -f "$FOLD/checkpoint_final.pth" "$FOLD/checkpoint_latest.pth"

echo '[R350] RAM full 30 epochs'
rm -rf "$RAM/full"
$PY/python scripts/train_r350_ram_unified_interaction_pcgrad.py \
  --dataset-root data/remote_variants/RAM-W600 \
  --prior-root outputs/priors/r323_ram_native_r317_single_seed \
  --r332-checkpoint "$R332" --output "$RAM/full" --epochs 30
test -s "$RAM/full/result.json"

echo '[R350] complete'
df -h /root/autodl-tmp

