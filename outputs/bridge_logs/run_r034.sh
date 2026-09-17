#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="TSRS_RSNA-Epiphysis"
CLEAN_DATASET="TSRS_RSNA-Epiphysis_clean_test_v2"
RUN_ID="r034_r025b_boundary_structure"
RUN_DIR="outputs/error_refiner/${DATASET}/${RUN_ID}"
CKPT="${RUN_DIR}/best_${RUN_ID}_sam_hqsam.pt"
SWEEP="${RUN_DIR}/threshold_sweep_val_${RUN_ID}.json"
OUTEXP="${RUN_ID}_sam_hqsam"

BASELINE_EXP="r025b_anchor_local_pixel_residual_sam_hqsam"
REFINED_EXP="allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam"
REFINED_EXP_2="hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9"

mkdir -p "${RUN_DIR}" outputs/analysis outputs/bridge_logs

python scripts/train_error_refiner.py \
  --dataset "${DATASET}" \
  --raw-root data/raw \
  --ablations-root outputs/ablations \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --refined-exp-2 "${REFINED_EXP_2}" \
  --target-mode allstack_anatomy_roi_twostage_boundary_refiner \
  --output "${CKPT}" \
  --img-size 512 \
  --epochs 72 \
  --min-epochs 18 \
  --patience 9 \
  --batch-size 4 \
  --base-channels 24 \
  --lr 4e-4 \
  --num-workers 4 \
  --aux-loss-weight 0.35 \
  --contrast-loss-weight 0.10 \
  --cldice-loss-weight 0.12 \
  --cldice-iters 4 \
  --instance-sep-loss-weight 0.20 \
  --stage2-boundary-loss-weight 0.45 \
  --gate-disagreement-threshold 0.08 \
  --gate-boundary-weight 0.85 \
  --gate-uncertainty-weight 0.35 \
  --gate-smooth-kernel 5 \
  --boundary-band-kernel 11 \
  --boundary-fg-weight 0.62 \
  --boundary-bg-weight 0.58 \
  --selection-dice-weight 0.45 \
  --selection-iou-weight 0.20 \
  --selection-precision-weight 0.15 \
  --selection-boundary-weight 0.20

python scripts/select_error_refiner_threshold.py \
  --dataset "${DATASET}" \
  --split val \
  --raw-root data/raw \
  --ablations-root outputs/ablations \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --refined-exp-2 "${REFINED_EXP_2}" \
  --checkpoint "${CKPT}" \
  --target-mode allstack_anatomy_roi_twostage_boundary_refiner \
  --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
  --selection-dice-weight 0.45 \
  --selection-iou-weight 0.20 \
  --selection-precision-weight 0.15 \
  --selection-boundary-weight 0.20 \
  --post-median-ksize 1 \
  --post-open-kernel 1 \
  --post-min-component 0 \
  --post-min-hole 0 \
  --output "${SWEEP}"

THR="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${SWEEP}").read_text())
print(data["best"]["threshold"])
PY
)"
echo "Selected threshold: ${THR}"

python scripts/infer_error_refiner.py \
  --dataset "${CLEAN_DATASET}" \
  --split test \
  --raw-root data/raw_variants \
  --ablations-root outputs/ablations_variants \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --refined-exp-2 "${REFINED_EXP_2}" \
  --checkpoint "${CKPT}" \
  --target-mode allstack_anatomy_roi_twostage_boundary_refiner \
  --output-name "${OUTEXP}" \
  --threshold "${THR}" \
  --post-median-ksize 1 \
  --post-open-kernel 1 \
  --post-min-component 0 \
  --post-min-hole 0

python scripts/evaluate_masks.py \
  --dataset "${CLEAN_DATASET}" \
  --raw-root data/raw_variants \
  --split test \
  --pred-dir "outputs/ablations_variants/${OUTEXP}/${CLEAN_DATASET}/test/masks" \
  --output "outputs/analysis/${RUN_ID}_clean_test_v2_metrics.json"

cat "outputs/analysis/${RUN_ID}_clean_test_v2_metrics.json"
