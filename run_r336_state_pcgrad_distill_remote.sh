#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src; PY=/root/miniconda3/bin/python; BIN=/root/miniconda3/bin
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"; DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
R317="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/$DATASET/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
RUN="$ROOT/outputs/nnunet/r336_state_pcgrad_distill"; RAM="$ROOT/outputs/ram_w600/r336_state_pcgrad_distill"
cd "$ROOT";mkdir -p outputs/bridge_logs "$RUN" "$RAM";exec > >(tee -a outputs/bridge_logs/r336_state_pcgrad_distill.log) 2>&1
export PATH="$BIN:$PATH" PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3361 nnUNet_compile=false nnUNet_n_proc_DA=0
echo "[R336] start $(date -Is)";df -h /root/autodl-tmp;nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
test -s outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth;test -s "$R317"
rm -rf "$RAM"
$PY scripts/train_r336_ram_gradient_surgery_distill.py --dataset-root data/remote_variants/RAM-W600 --prior-root outputs/priors/r323_ram_native_r317_single_seed --r332-checkpoint outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth --output "$RAM" --epochs 18 --steps 2
test -s "$RAM/result.json"

export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"
export R336_TSRS_INIT_CHECKPOINT="$R317" R336_TSRS_EPOCHS=18 R336_TSRS_SEAM_WEIGHT=.020 R336_TSRS_DISTILL_WEIGHT=.20
TRAINER_DIR="$($PY -c "from pathlib import Path;import nnunetv2;print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/nnUNetTrainerR336StatePCGradDistill.py "$TRAINER_DIR/nnUNetTrainerR336StatePCGradDistill.py";$PY -m py_compile "$TRAINER_DIR/nnUNetTrainerR336StatePCGradDistill.py"
test -d "$nnUNet_preprocessed/$DATASET/nnUNetPlans_2d";rm -rf "$nnUNet_results"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR336StatePCGradDistill
FOLD="$nnUNet_results/$DATASET/nnUNetTrainerR336StatePCGradDistill__nnUNetPlans__2d/fold_0";test -s "$FOLD/checkpoint_best.pth"
PRED="$RUN/predictions_val";MASKS="$RUN/masks_val";rm -rf "$PRED" "$MASKS"
nnUNetv2_predict -i outputs/nnunet/r317_image_centers_only/imagesValPrior -o "$PRED" -d 204 -c 2d -f 0 -tr nnUNetTrainerR336StatePCGradDistill -chk checkpoint_best.pth
$PY scripts/convert_nnunet_predictions.py --pred-dir "$PRED" --mask-dir "$MASKS" --expected-count 96 --strip-prefix val_
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$MASKS" --output outputs/analysis/r336_tsrs_state_pcgrad_distill_original_val_r201.json --expected-count 96
rm -rf "$PRED";rm -f "$FOLD/checkpoint_final.pth" "$FOLD/checkpoint_latest.pth"
echo "[R336] done $(date -Is)";df -h /root/autodl-tmp
