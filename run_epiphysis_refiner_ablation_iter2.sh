#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"
PRED_ROOT="outputs/ablations"

BASELINE_EXP="zero_shot_box_only_conf020_pad005"
ITER2_REFINED_EXP="zero_shot_mask_to_prompt_refine_iter2_pad003_neg2_k5"
ITER2_ERROR_REFINER_EXP="error_refiner_iter2_unet512_b32"
ITER2_ERROR_REFINER_CKPT="outputs/error_refiner/${DATASET}/best_iter2.pt"
REFINE_PROMPT_SCHEME="${REFINE_PROMPT_SCHEME:-box+point+mask}"

run_infer_eval() {
  local split="$1"
  local exp_name="$2"
  shift 2
  local pred_out_root="${PRED_ROOT}/${exp_name}"

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --out-root "${pred_out_root}" \
    "$@"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${split}" \
    --pred-dir "${pred_out_root}/${DATASET}/${split}/masks"
}

python scripts/download_sam_checkpoint.py

run_infer_eval test "${BASELINE_EXP}"

for split in train val test; do
  run_infer_eval \
    "${split}" \
    "${ITER2_REFINED_EXP}" \
    --mask-to-prompt-refine \
    --refine-box-padding-ratio 0.03 \
    --refine-num-positive-points 1 \
    --refine-num-negative-points 2 \
    --refine-negative-dilate-kernel 5 \
    --refine-iters 2 \
    --refine-prompt-scheme "${REFINE_PROMPT_SCHEME}"
done

python scripts/train_error_refiner.py \
  --dataset "${DATASET}" \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${ITER2_REFINED_EXP}" \
  --img-size 512 \
  --epochs 80 \
  --min-epochs 20 \
  --patience 10 \
  --batch-size 8 \
  --lr 5e-4 \
  --num-workers 8 \
  --base-channels 32 \
  --output "${ITER2_ERROR_REFINER_CKPT}"

python scripts/select_error_refiner_threshold.py \
  --dataset "${DATASET}" \
  --split val \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${ITER2_REFINED_EXP}" \
  --checkpoint "${ITER2_ERROR_REFINER_CKPT}" \
  --thresholds "0.40,0.45,0.50,0.55,0.60" \
  --output "outputs/error_refiner/${DATASET}/threshold_sweep_val_iter2.json"

SELECTED_THRESHOLD="$(python - <<'PY'
import json
from pathlib import Path
path = Path("outputs/error_refiner/TSRS_RSNA-Epiphysis/threshold_sweep_val_iter2.json")
data = json.loads(path.read_text())
print(data["best"]["threshold"])
PY
)"

python scripts/infer_error_refiner.py \
  --dataset "${DATASET}" \
  --split test \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${ITER2_REFINED_EXP}" \
  --checkpoint "${ITER2_ERROR_REFINER_CKPT}" \
  --output-name "${ITER2_ERROR_REFINER_EXP}" \
  --threshold "${SELECTED_THRESHOLD}"

python scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --split test \
  --pred-dir "${PRED_ROOT}/${ITER2_ERROR_REFINER_EXP}/${DATASET}/test/masks"

python scripts/summarize_ablation_metrics.py \
  --dataset "${DATASET}" \
  --split test \
  --experiments \
    "${BASELINE_EXP}" \
    "${ITER2_REFINED_EXP}" \
    "${ITER2_ERROR_REFINER_EXP}" \
  --output "${PRED_ROOT}/${DATASET}_test_refiner_path_iter2_summary.md"
