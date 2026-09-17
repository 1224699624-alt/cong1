#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="TSRS_RSNA-Epiphysis"
PRED_ROOT="outputs/ablations"
ERROR_ROOT="outputs/error_refiner/${DATASET}"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"
IMG_SIZE="${IMG_SIZE:-768}"

BASELINE_EXP="${BASELINE_EXP:-zero_shot_box_only_conf020_pad005}"
REFINED_EXP="${REFINED_EXP:-zero_shot_mask_to_prompt_refine_pad003_neg2_k5}"

REFINER_EXP="${REFINER_EXP:-single_candidate_sam_roi_filtered_refiner}"
REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-${ERROR_ROOT}/best_single_candidate_sam_roi_filtered_refiner.pt}"
THRESHOLD_JSON="${THRESHOLD_JSON:-${ERROR_ROOT}/threshold_sweep_val_single_candidate_sam_roi_filtered_refiner.json}"
THRESHOLDS="${THRESHOLDS:-0.44,0.46,0.48,0.50,0.52,0.54}"

run_candidate_generation() {
  local split="$1"

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --imgsz "${IMG_SIZE}" \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --out-root "outputs/ablations/${BASELINE_EXP}"

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --imgsz "${IMG_SIZE}" \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --mask-to-prompt-refine \
    --refine-box-padding-ratio 0.03 \
    --refine-num-positive-points 1 \
    --refine-num-negative-points 2 \
    --refine-negative-dilate-kernel 5 \
    --out-root "outputs/ablations/${REFINED_EXP}"
}

python scripts/download_sam_checkpoint.py

run_candidate_generation train
run_candidate_generation val
run_candidate_generation test

python scripts/train_error_refiner.py \
  --dataset "${DATASET}" \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --target-mode allstack_anatomy_roi_only \
  --img-size 512 \
  --epochs 80 \
  --min-epochs 20 \
  --patience 10 \
  --batch-size 8 \
  --lr 5e-4 \
  --num-workers 8 \
  --base-channels 32 \
  --aux-loss-weight 0.35 \
  --contrast-loss-weight 0.10 \
  --gate-disagreement-threshold 0.10 \
  --gate-boundary-weight 0.75 \
  --gate-uncertainty-weight 0.35 \
  --gate-smooth-kernel 5 \
  --selection-dice-weight 0.30 \
  --selection-iou-weight 0.25 \
  --selection-precision-weight 0.25 \
  --selection-boundary-weight 0.20 \
  --output "${REFINER_CHECKPOINT}"

python scripts/select_error_refiner_threshold.py \
  --dataset "${DATASET}" \
  --split val \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --checkpoint "${REFINER_CHECKPOINT}" \
  --thresholds "${THRESHOLDS}" \
  --selection-dice-weight 0.30 \
  --selection-iou-weight 0.25 \
  --selection-precision-weight 0.25 \
  --selection-boundary-weight 0.20 \
  --output "${THRESHOLD_JSON}"

SELECTED_THRESHOLD="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${THRESHOLD_JSON}").read_text())
print(data["best"]["threshold"])
PY
)"

python scripts/infer_error_refiner.py \
  --dataset "${DATASET}" \
  --split test \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --checkpoint "${REFINER_CHECKPOINT}" \
  --output-name "${REFINER_EXP}" \
  --threshold "${SELECTED_THRESHOLD}"

python scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --split test \
  --pred-dir "${PRED_ROOT}/${REFINER_EXP}/${DATASET}/test/masks"

python scripts/summarize_ablation_metrics.py \
  --dataset "${DATASET}" \
  --split test \
  --experiments \
    "${BASELINE_EXP}" \
    "${REFINED_EXP}" \
    "${REFINER_EXP}" \
  --output "${PRED_ROOT}/${DATASET}_test_single_candidate_sam_roi_filtered_summary.md"
