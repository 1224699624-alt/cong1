#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RAW_ROOT="${RAW_ROOT:-data/raw_variants}"
OUT_ROOT="${OUT_ROOT:-outputs/reannotation_quality_compare}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"
HQ_SAM_CHECKPOINT="${HQ_SAM_CHECKPOINT:-checkpoints/sam_hq_vit_b.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"
IMG_SIZE="${IMG_SIZE:-1024}"

OLD_DATASET="${OLD_DATASET:-TSRS_RSNA-Epiphysis_badlabel_filtered_v1}"
NEW_DATASET="${NEW_DATASET:-TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1}"

BASELINE_EXP="${BASELINE_EXP:-zero_shot_box_only_conf020_pad005}"
REFINED_EXP="${REFINED_EXP:-zero_shot_mask_to_prompt_refine_pad003_neg2_k5}"
REFINED_EXP_2="${REFINED_EXP_2:-hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9}"
REFINER_MODE="${REFINER_MODE:-allstack_anatomy_roi_boundary_prefgate_trimfirst}"
REFINER_TAG="${REFINER_TAG:-boundary_prefgate_trimfirst}"
THRESHOLDS="${THRESHOLDS:-0.42,0.44,0.46,0.48,0.50,0.52}"

run_candidate_if_missing() {
  local dataset="$1"
  local split="$2"
  local exp_name="$3"
  shift 3
  local mask_dir="${OUT_ROOT}/${exp_name}/${dataset}/${split}/masks"
  local expected_count
  local current_count
  expected_count="$(find "${RAW_ROOT}/${dataset}/${split}" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.bmp' \) | wc -l)"
  current_count="0"
  if [[ -d "${mask_dir}" ]]; then
    current_count="$(find "${mask_dir}" -maxdepth 1 -name '*.png' | wc -l)"
  fi
  if [[ "${current_count}" -ge "${expected_count}" && "${expected_count}" -gt 0 ]]; then
    echo "[skip] ${exp_name}/${dataset}/${split} already has ${current_count}/${expected_count} masks"
    return
  fi
  echo "[run] ${exp_name}/${dataset}/${split} has ${current_count}/${expected_count} masks"

  python scripts/infer_yolo_sam.py \
    --dataset "${dataset}" \
    --raw-root "${RAW_ROOT}" \
    --split "${split}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --imgsz "${IMG_SIZE}" \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --out-root "${OUT_ROOT}/${exp_name}" \
    "$@"
}

run_eval() {
  local dataset="$1"
  local split="$2"
  local exp_name="$3"
  python scripts/evaluate_masks.py \
    --dataset "${dataset}" \
    --raw-root "${RAW_ROOT}" \
    --split "${split}" \
    --pred-dir "${OUT_ROOT}/${exp_name}/${dataset}/${split}/masks"
}

run_dataset() {
  local dataset="$1"
  local checkpoint="${OUT_ROOT}/error_refiner/${dataset}/best_${REFINER_TAG}.pt"
  local threshold_json="${OUT_ROOT}/error_refiner/${dataset}/threshold_sweep_val_${REFINER_TAG}.json"
  local refiner_exp="${REFINER_TAG}_refiner"

  echo "==== Dataset: ${dataset} ===="
  for split in train val test; do
    run_candidate_if_missing "${dataset}" "${split}" "${BASELINE_EXP}" \
      --sam-checkpoint "${SAM_CHECKPOINT}"
    run_eval "${dataset}" "${split}" "${BASELINE_EXP}"

    run_candidate_if_missing "${dataset}" "${split}" "${REFINED_EXP}" \
      --sam-checkpoint "${SAM_CHECKPOINT}" \
      --mask-to-prompt-refine \
      --refine-box-padding-ratio 0.03 \
      --refine-num-positive-points 1 \
      --refine-num-negative-points 2 \
      --refine-negative-dilate-kernel 5
    run_eval "${dataset}" "${split}" "${REFINED_EXP}"

    run_candidate_if_missing "${dataset}" "${split}" "${REFINED_EXP_2}" \
      --sam-backend hq_sam \
      --sam-checkpoint "${HQ_SAM_CHECKPOINT}" \
      --sam-model-type "${SAM_MODEL_TYPE}" \
      --mask-to-prompt-refine \
      --refine-box-padding-ratio 0.03 \
      --refine-num-positive-points 1 \
      --refine-num-negative-points 2 \
      --refine-negative-dilate-kernel 9 \
      --refine-iters 2 \
      --refine-prompt-scheme "box+point+mask"
    run_eval "${dataset}" "${split}" "${REFINED_EXP_2}"
  done

  python scripts/train_error_refiner.py \
    --dataset "${dataset}" \
    --raw-root "${RAW_ROOT}" \
    --ablations-root "${OUT_ROOT}" \
    --baseline-exp "${BASELINE_EXP}" \
    --refined-exp "${REFINED_EXP}" \
    --refined-exp-2 "${REFINED_EXP_2}" \
    --target-mode "${REFINER_MODE}" \
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
    --selector-loss-weight 0.25 \
    --gate-disagreement-threshold 0.10 \
    --gate-boundary-weight 0.75 \
    --gate-uncertainty-weight 0.35 \
    --gate-smooth-kernel 5 \
    --boundary-fg-weight 0.55 \
    --boundary-bg-weight 0.62 \
    --selection-dice-weight 0.25 \
    --selection-iou-weight 0.20 \
    --selection-precision-weight 0.25 \
    --selection-boundary-weight 0.30 \
    --output "${checkpoint}"

  python scripts/select_error_refiner_threshold.py \
    --dataset "${dataset}" \
    --raw-root "${RAW_ROOT}" \
    --ablations-root "${OUT_ROOT}" \
    --split val \
    --baseline-exp "${BASELINE_EXP}" \
    --refined-exp "${REFINED_EXP}" \
    --refined-exp-2 "${REFINED_EXP_2}" \
    --checkpoint "${checkpoint}" \
    --thresholds "${THRESHOLDS}" \
    --selection-dice-weight 0.25 \
    --selection-iou-weight 0.20 \
    --selection-precision-weight 0.25 \
    --selection-boundary-weight 0.30 \
    --output "${threshold_json}"

  local selected_threshold
  selected_threshold="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${threshold_json}").read_text())
print(data["best"]["threshold"])
PY
)"
  echo "Selected threshold for ${dataset}: ${selected_threshold}"

  python scripts/infer_error_refiner.py \
    --dataset "${dataset}" \
    --raw-root "${RAW_ROOT}" \
    --ablations-root "${OUT_ROOT}" \
    --split test \
    --baseline-exp "${BASELINE_EXP}" \
    --refined-exp "${REFINED_EXP}" \
    --refined-exp-2 "${REFINED_EXP_2}" \
    --target-mode "${REFINER_MODE}" \
    --checkpoint "${checkpoint}" \
    --output-name "${refiner_exp}" \
    --threshold "${selected_threshold}"

  python scripts/evaluate_masks.py \
    --dataset "${dataset}" \
    --raw-root "${RAW_ROOT}" \
    --split test \
    --pred-dir "${OUT_ROOT}/${refiner_exp}/${dataset}/test/masks"

  python scripts/summarize_ablation_metrics.py \
    --dataset "${dataset}" \
    --split test \
    --ablations-root "${OUT_ROOT}" \
    --experiments "${BASELINE_EXP}" "${REFINED_EXP}" "${REFINED_EXP_2}" "${refiner_exp}" \
    --output "${OUT_ROOT}/${dataset}_test_summary.md"
}

python scripts/download_sam_checkpoint.py

run_dataset "${OLD_DATASET}"
run_dataset "${NEW_DATASET}"

python scripts/compare_reannotation_quality.py \
  --old-dataset "${OLD_DATASET}" \
  --new-dataset "${NEW_DATASET}" \
  --ablations-root "${OUT_ROOT}" \
  --refiner-exp "${REFINER_TAG}_refiner" \
  --output "${OUT_ROOT}/old_vs_new_reannotation_quality.md"
