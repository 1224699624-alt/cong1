#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"
BASELINE_EXP="zero_shot_box_only_conf020_pad005"
REFINED_EXP="zero_shot_mask_to_prompt_refine_pad003_neg2_k5"
REFINER_EXP="error_refiner_oldrefine_tuned"
REFINER_CHECKPOINT="outputs/error_refiner/${DATASET}/best_oldrefine_tuned.pt"

run_candidate_generation() {
  local split="$1"

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --out-root "outputs/ablations/${BASELINE_EXP}"

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --mask-to-prompt-refine \
    --refine-box-padding-ratio 0.03 \
    --refine-num-positive-points 1 \
    --refine-num-negative-points 2 \
    --refine-negative-dilate-kernel 5 \
    --refine-iters 2 \
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
  --img-size 512 \
  --epochs 80 \
  --min-epochs 20 \
  --patience 10 \
  --batch-size 8 \
  --lr 5e-4 \
  --num-workers 8 \
  --base-channels 32 \
  --output "${REFINER_CHECKPOINT}"

python scripts/select_error_refiner_threshold.py \
  --dataset "${DATASET}" \
  --split val \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --checkpoint "${REFINER_CHECKPOINT}" \
  --thresholds "0.40,0.45,0.50,0.55,0.60" \
  --output "outputs/error_refiner/${DATASET}/threshold_sweep_val_oldrefine_tuned.json"

SELECTED_THRESHOLD="$(python - <<'PY'
import json
from pathlib import Path
path = Path("outputs/error_refiner/TSRS_RSNA-Epiphysis/threshold_sweep_val_oldrefine_tuned.json")
data = json.loads(path.read_text())
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
  --pred-dir "outputs/ablations/${REFINER_EXP}/${DATASET}/test/masks"
