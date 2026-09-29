#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
WORK=/root/r351_workspace
PY=/root/miniconda3/bin
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
TRAINER=nnUNetTrainerR351DualPrior
MODE="${1:-audit}"
cd "$ROOT"
mkdir -p "$WORK/logs"
exec > >(tee "$WORK/logs/tsrs_${MODE}.log") 2>&1
export PATH="$PY:$PATH" PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 nnUNet_compile=false nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$WORK/tsrs_${MODE}/nnUNet_results"
export R351_TSRS_INIT="$ROOT/outputs/nnunet/r350_unified_interaction_pcgrad/nnUNet_results/$DATASET/nnUNetTrainerR350UnifiedInteractionPCGrad__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
export R351_RAM_INIT="$ROOT/outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth"
export R351_OVERLAP_CHECKPOINT="${R351_OVERLAP_CHECKPOINT:-$WORK/overlap/full/best.pth}"
export R351_EPOCHS=2 R351_ITERATIONS=100
test -s "$R351_TSRS_INIT"
test -s "$R351_RAM_INIT"
test -s "$R351_OVERLAP_CHECKPOINT"
test ! -d "$nnUNet_results/$DATASET"
TRAINER_DIR="$($PY/python -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp "scripts/nnunet_trainers/$TRAINER.py" "$TRAINER_DIR/$TRAINER.py"
if [ "$MODE" = audit ]; then export R351_AUDIT_ONLY=true; else unset R351_AUDIT_ONLY || true; fi
nnUNetv2_train 204 2d 0 -tr "$TRAINER"
FOLD="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__2d/fold_0"
if [ "$MODE" = audit ]; then
  "$PY/python" -c "import json; assert json.load(open('$FOLD/smoke_audit.json'))['passed']"
  exit 0
fi
test -s "$FOLD/checkpoint_best.pth"
PRED="$WORK/tsrs_${MODE}/predictions_best"
MASKS="$WORK/tsrs_${MODE}/masks_best"
nnUNetv2_predict -i outputs/nnunet/r317_image_centers_only/imagesValPrior -o "$PRED" \
  -d 204 -c 2d -f 0 -tr "$TRAINER" -chk checkpoint_best.pth
"$PY/python" scripts/convert_nnunet_predictions.py --pred-dir "$PRED" --mask-dir "$MASKS" --expected-count 96 --strip-prefix val_
"$PY/python" scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$MASKS" --output "$WORK/tsrs_${MODE}/r201.json" --expected-count 96
"$PY/python" scripts/summarize_r351_tsrs.py --run "$WORK/tsrs_${MODE}" --checkpoint "$FOLD/checkpoint_best.pth"
echo '[R351 TSRS] complete'
