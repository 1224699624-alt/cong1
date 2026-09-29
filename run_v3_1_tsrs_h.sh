#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
CODE=/root/v4_1/code
RUN=/root/v4_1/tsrs/H_input
test ! -e "$RUN"
mkdir -p "$RUN"
exec > >(tee "$RUN/run.log") 2>&1
cd "$ROOT"
export PATH=/root/miniconda3/bin:$PATH PYTHONPATH="$CODE:/root/v3_round3/code:$ROOT/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 nnUNet_compile=false nnUNet_n_proc_DA=0
export V3_ARM=B V3_PROTOCOL=/root/v3_round3/ROUND3_PROTOCOL_LOCK.json V31_PROTOCOL=/root/v4_1/R4_1_PROTOCOL_LOCK.json
export V3_NATIVE="$ROOT/outputs/nnunet/r317_image_centers_only/r202_two_channel_zero_prior.pth"
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"
TDIR="$(python -c "from pathlib import Path;import nnunetv2;print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp "$CODE/nnUNetTrainerV31HeatmapInput.py" "$TDIR/nnUNetTrainerV31HeatmapInput.py"
cp /root/v3_round3/code/nnUNetTrainerV3Round3.py "$TDIR/nnUNetTrainerV3Round3.py"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerV31HeatmapInput
FOLD="$nnUNet_results/$DATASET/nnUNetTrainerV31HeatmapInput__nnUNetPlans__2d/fold_0"
nnUNetv2_predict -i "$ROOT/outputs/nnunet/r317_image_centers_only/imagesValPrior" -o "$RUN/predictions" -d 204 -c 2d -f 0 -tr nnUNetTrainerV31HeatmapInput -chk checkpoint_best.pth
python /root/v3_round3/code/evaluate_round3_tsrs.py --gt "$ROOT/data/raw/TSRS_RSNA-Epiphysis/val_labels" --pred "$RUN/predictions" --checkpoint "$FOLD/checkpoint_best.pth" --evaluator-source "$ROOT/scripts/run_r201_unified_eval.py" --output "$RUN/r201.json"
