#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
HQ_SAM_CHECKPOINT="${HQ_SAM_CHECKPOINT:-checkpoints/sam_hq_vit_b.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"
PRED_ROOT="outputs/ablations"
ERROR_ROOT="outputs/error_refiner/${DATASET}"

BASELINE_EXP="hq_sam_box_only_conf020_pad005"
REFINED_EXP="hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9"
REFINER_EXP="hq_sam_error_refiner_iter2_pad003_neg2_k9"
REFINER_CHECKPOINT="${ERROR_ROOT}/best_hq_sam_iter2_pad003_neg2_k9.pt"
THRESHOLD_JSON="${ERROR_ROOT}/threshold_sweep_val_hq_sam_iter2_pad003_neg2_k9.json"
THRESHOLDS="${THRESHOLDS:-0.40,0.45,0.50,0.55,0.60}"
REFINER_THRESHOLD_OVERRIDE="${REFINER_THRESHOLD_OVERRIDE:-}"

run_baseline_eval() {
  local split="$1"
  local pred_dir="${PRED_ROOT}/${BASELINE_EXP}/${DATASET}/${split}/masks"
  if [[ -d "${pred_dir}" ]] && [[ -n "$(find "${pred_dir}" -maxdepth 1 -name '*.png' -print -quit)" ]]; then
    echo "HQ-SAM baseline already exists for ${split}: ${pred_dir}"
    return
  fi

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-backend hq_sam \
    --sam-checkpoint "${HQ_SAM_CHECKPOINT}" \
    --sam-model-type "${SAM_MODEL_TYPE}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --out-root "${PRED_ROOT}/${BASELINE_EXP}"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --pred-dir "${pred_dir}"
}

run_refined_eval() {
  local split="$1"
  local pred_dir="${PRED_ROOT}/${REFINED_EXP}/${DATASET}/${split}/masks"
  if [[ -d "${pred_dir}" ]] && [[ -n "$(find "${pred_dir}" -maxdepth 1 -name '*.png' -print -quit)" ]]; then
    echo "HQ-SAM refined candidate already exists for ${split}: ${pred_dir}"
    return
  fi

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-backend hq_sam \
    --sam-checkpoint "${HQ_SAM_CHECKPOINT}" \
    --sam-model-type "${SAM_MODEL_TYPE}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --mask-to-prompt-refine \
    --refine-box-padding-ratio 0.03 \
    --refine-num-positive-points 1 \
    --refine-num-negative-points 2 \
    --refine-negative-dilate-kernel 9 \
    --refine-iters 2 \
    --refine-prompt-scheme "box+point+mask" \
    --out-root "${PRED_ROOT}/${REFINED_EXP}"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --pred-dir "${pred_dir}"
}

for split in train val test; do
  run_baseline_eval "${split}"
  run_refined_eval "${split}"
done

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
  --aux-loss-weight 0.35 \
  --output "${REFINER_CHECKPOINT}"

python scripts/select_error_refiner_threshold.py \
  --dataset "${DATASET}" \
  --split val \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --checkpoint "${REFINER_CHECKPOINT}" \
  --thresholds "${THRESHOLDS}" \
  --output "${THRESHOLD_JSON}"

if [[ -n "${REFINER_THRESHOLD_OVERRIDE}" ]]; then
  SELECTED_THRESHOLD="${REFINER_THRESHOLD_OVERRIDE}"
else
  SELECTED_THRESHOLD="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${THRESHOLD_JSON}").read_text())
print(data["best"]["threshold"])
PY
)"
fi

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
  --output "${PRED_ROOT}/${DATASET}_test_hq_sam_error_refiner_summary.md"

