#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
CODE=/root/v3_round3/code
WORK=/root/v3_round3/tsrs
MODE="${1:?N0 A or B}"
case "$MODE" in N0|A|B) ;; *) exit 2;; esac
RUN="$WORK/$MODE"
test ! -e "$RUN"
mkdir -p "$RUN"
exec > >(tee "$RUN/run.log") 2>&1
cd "$ROOT"
export PATH=/root/miniconda3/bin:$PATH PYTHONPATH="$CODE:$ROOT/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 nnUNet_compile=false nnUNet_n_proc_DA=0
export V3_ARM="$MODE" V3_PROTOCOL=/root/v3_round3/ROUND3_PROTOCOL_LOCK.json
export V3_NATIVE="$ROOT/outputs/nnunet/r317_image_centers_only/r202_two_channel_zero_prior.pth"
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"
python "$CODE/round3_common.py" --root "$ROOT" --protocol "$V3_PROTOCOL" >/dev/null
TRAINER=nnUNetTrainerV3Round3
TDIR="$(python -c "from pathlib import Path;import nnunetv2;print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp "$CODE/$TRAINER.py" "$TDIR/$TRAINER.py"
if [ "$MODE" = N0 ]; then
  TRAINER=nnUNetTrainer
  FOLD="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__2d/fold_0"
  mkdir -p "$FOLD"
  SOURCE="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/$DATASET/nnUNetTrainerR317Continuous035__nnUNetPlans__2d"
  ln -s "$SOURCE/dataset.json" "$(dirname "$FOLD")/dataset.json"
  ln -s "$SOURCE/plans.json" "$(dirname "$FOLD")/plans.json"
  ln -s "$V3_NATIVE" "$FOLD/checkpoint_best.pth"
else
  nnUNetv2_train 204 2d 0 -tr "$TRAINER"
  FOLD="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__2d/fold_0"
fi
nnUNetv2_predict -i "$ROOT/outputs/nnunet/r317_image_centers_only/imagesValPrior" -o "$RUN/predictions" -d 204 -c 2d -f 0 -tr "$TRAINER" -chk checkpoint_best.pth
python "$CODE/evaluate_round3_tsrs.py" --gt "$ROOT/data/raw/TSRS_RSNA-Epiphysis/val_labels" --pred "$RUN/predictions" --checkpoint "$FOLD/checkpoint_best.pth" --evaluator-source "$ROOT/scripts/run_r201_unified_eval.py" --output "$RUN/r201.json"
