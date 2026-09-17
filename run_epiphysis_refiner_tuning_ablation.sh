#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"
PRED_ROOT="outputs/ablations"
ERROR_ROOT="outputs/error_refiner/${DATASET}"

BASELINE_EXP="zero_shot_box_only_conf020_pad005"
THRESHOLDS="${THRESHOLDS:-0.40,0.45,0.50,0.55,0.60}"

run_baseline_if_missing() {
  local split="$1"
  local pred_dir="${PRED_ROOT}/${BASELINE_EXP}/${DATASET}/${split}/masks"
  if [[ -d "${pred_dir}" ]] && [[ -n "$(find "${pred_dir}" -maxdepth 1 -name '*.png' -print -quit)" ]]; then
    echo "Baseline already exists for ${split}: ${pred_dir}"
    return
  fi

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
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

run_refine_eval() {
  local split="$1"
  local exp_name="$2"
  local refine_box_padding_ratio="$3"
  local refine_num_positive_points="$4"
  local refine_num_negative_points="$5"
  local refine_negative_dilate_kernel="$6"

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
    --refine-iters 2 \
    --refine-prompt-scheme "box+point+mask" \
    --refine-box-padding-ratio "${refine_box_padding_ratio}" \
    --refine-num-positive-points "${refine_num_positive_points}" \
    --refine-num-negative-points "${refine_num_negative_points}" \
    --refine-negative-dilate-kernel "${refine_negative_dilate_kernel}" \
    --out-root "${PRED_ROOT}/${exp_name}"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --pred-dir "${PRED_ROOT}/${exp_name}/${DATASET}/${split}/masks"
}

run_error_refiner_variant() {
  local refine_exp="$1"
  local refiner_exp="$2"
  local checkpoint_name="$3"
  local threshold_json_name="$4"
  local lr="$5"
  local base_channels="$6"
  local aux_loss_weight="$7"

  local checkpoint_path="${ERROR_ROOT}/${checkpoint_name}"
  local threshold_json_path="${ERROR_ROOT}/${threshold_json_name}"

  python scripts/train_error_refiner.py \
    --dataset "${DATASET}" \
    --baseline-exp "${BASELINE_EXP}" \
    --refined-exp "${refine_exp}" \
    --img-size 512 \
    --epochs 80 \
    --min-epochs 20 \
    --patience 10 \
    --batch-size 8 \
    --lr "${lr}" \
    --num-workers 8 \
    --base-channels "${base_channels}" \
    --aux-loss-weight "${aux_loss_weight}" \
    --output "${checkpoint_path}"

  python scripts/select_error_refiner_threshold.py \
    --dataset "${DATASET}" \
    --split val \
    --baseline-exp "${BASELINE_EXP}" \
    --refined-exp "${refine_exp}" \
    --checkpoint "${checkpoint_path}" \
    --thresholds "${THRESHOLDS}" \
    --output "${threshold_json_path}"

  local selected_threshold
  selected_threshold="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${threshold_json_path}").read_text())
print(data["best"]["threshold"])
PY
)"

  python scripts/infer_error_refiner.py \
    --dataset "${DATASET}" \
    --split test \
    --baseline-exp "${BASELINE_EXP}" \
    --refined-exp "${refine_exp}" \
    --checkpoint "${checkpoint_path}" \
    --output-name "${refiner_exp}" \
    --threshold "${selected_threshold}"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split test \
    --pred-dir "${PRED_ROOT}/${refiner_exp}/${DATASET}/test/masks"
}

run_full_variant() {
  local refine_exp="$1"
  local refiner_exp="$2"
  local checkpoint_name="$3"
  local threshold_json_name="$4"
  local refine_box_padding_ratio="$5"
  local refine_num_positive_points="$6"
  local refine_num_negative_points="$7"
  local refine_negative_dilate_kernel="$8"
  local lr="$9"
  local base_channels="${10}"
  local aux_loss_weight="${11}"

  for split in train val test; do
    run_refine_eval \
      "${split}" \
      "${refine_exp}" \
      "${refine_box_padding_ratio}" \
      "${refine_num_positive_points}" \
      "${refine_num_negative_points}" \
      "${refine_negative_dilate_kernel}"
  done

  run_error_refiner_variant \
    "${refine_exp}" \
    "${refiner_exp}" \
    "${checkpoint_name}" \
    "${threshold_json_name}" \
    "${lr}" \
    "${base_channels}" \
    "${aux_loss_weight}"
}

python scripts/download_sam_checkpoint.py

for split in train val test; do
  run_baseline_if_missing "${split}"
done

# Variant A: stronger positive anchoring in multi-prompt refinement.
run_full_variant \
  "zero_shot_refine_iter2_boxpointmask_pad003_pos2_neg2_k5" \
  "error_refiner_tuned_pos2_neg2_k5" \
  "best_tuned_pos2_neg2_k5.pt" \
  "threshold_sweep_val_tuned_pos2_neg2_k5.json" \
  "0.03" \
  "2" \
  "2" \
  "5" \
  "5e-4" \
  "32" \
  "0.35"

# Variant B: same front-end candidate, lighter auxiliary loss.
run_error_refiner_variant \
  "zero_shot_refine_iter2_boxpointmask_pad003_pos2_neg2_k5" \
  "error_refiner_tuned_pos2_aux020" \
  "best_tuned_pos2_aux020.pt" \
  "threshold_sweep_val_tuned_pos2_aux020.json" \
  "5e-4" \
  "32" \
  "0.20"

# Variant C: same front-end candidate, smaller model and lower LR.
run_error_refiner_variant \
  "zero_shot_refine_iter2_boxpointmask_pad003_pos2_neg2_k5" \
  "error_refiner_tuned_pos2_bc24_lr3e4" \
  "best_tuned_pos2_bc24_lr3e4.pt" \
  "threshold_sweep_val_tuned_pos2_bc24_lr3e4.json" \
  "3e-4" \
  "24" \
  "0.35"

python scripts/summarize_ablation_metrics.py \
  --dataset "${DATASET}" \
  --split test \
  --experiments \
    "${BASELINE_EXP}" \
    "zero_shot_refine_iter2_boxpointmask_pad003_pos2_neg2_k5" \
    "error_refiner_tuned_pos2_neg2_k5" \
    "error_refiner_tuned_pos2_aux020" \
    "error_refiner_tuned_pos2_bc24_lr3e4" \
  --output "${PRED_ROOT}/${DATASET}_test_refiner_tuning_summary.md"
