#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
WORK=/root/r351_workspace
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
TRAINER=nnUNetTrainerR351DualPrior
RUN="$WORK/tsrs_positive_guard"
test ! -d "$RUN"
mkdir -p "$RUN"
exec > >(tee "$WORK/logs/tsrs_positive_guard.log") 2>&1
cd "$ROOT"
export PATH=/root/miniconda3/bin:$PATH PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 nnUNet_compile=false PYTHONUNBUFFERED=1
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$WORK/tsrs_pilot/nnUNet_results" R351_POSITIVE_GATE=1
export R351_RAM_INIT="$ROOT/outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth"
export R351_OVERLAP_CHECKPOINT="$WORK/overlap/full/best.pth"
test -s "$R351_RAM_INIT"
test -s "$R351_OVERLAP_CHECKPOINT"
export R351_TSRS_INIT="$ROOT/outputs/nnunet/r350_unified_interaction_pcgrad/nnUNet_results/$DATASET/nnUNetTrainerR350UnifiedInteractionPCGrad__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
TRAINER_DIR="$(python -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/$TRAINER.py "$TRAINER_DIR/$TRAINER.py"
CKPT="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
test -s "$CKPT"
nnUNetv2_predict -i outputs/nnunet/r317_image_centers_only/imagesValPrior -o "$RUN/predictions" \
  -d 204 -c 2d -f 0 -tr "$TRAINER" -chk checkpoint_best.pth
python scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions" --mask-dir "$RUN/masks" --expected-count 96 --strip-prefix val_
python scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$RUN/masks" --output "$RUN/r201.json" --expected-count 96
python scripts/summarize_r351_tsrs.py --run "$RUN" --checkpoint "$CKPT" --inference-policy positive_seam_guard
echo '[R351 TSRS positive guard diagnostic] complete'
