#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"
MEDSAM_CHECKPOINT="${MEDSAM_CHECKPOINT:-checkpoints/medsam_vit_b.pth}"
DATASET="TSRS_RSNA-Epiphysis"
SPLIT="test"
FINE_TUNE_ROOT="outputs/sam_finetune/${DATASET}/ablations"
PRED_ROOT="outputs/ablations"

mkdir -p "${FINE_TUNE_ROOT}" "${PRED_ROOT}"

run_zero_shot() {
  local exp_name="$1"
  local infer_extra_args="${2:-}"
  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --save-overlays \
    --out-root "${PRED_ROOT}/${exp_name}" \
    ${infer_extra_args}

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --pred-dir "${PRED_ROOT}/${exp_name}/${DATASET}/${SPLIT}/masks"
}

run_medsam_zero_shot() {
  local exp_name="$1"
  local pred_out_root="${PRED_ROOT}/${exp_name}"

  python scripts/infer_yolo_medsam.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --medsam-checkpoint "${MEDSAM_CHECKPOINT}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --save-overlays \
    --out-root "${pred_out_root}"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --pred-dir "${pred_out_root}/${DATASET}/${SPLIT}/masks"
}

run_finetune_ablation() {
  local exp_name="$1"
  local train_extra_args="$2"
  local infer_extra_args="${3:-}"

  local finetuned_checkpoint="${FINE_TUNE_ROOT}/${exp_name}.pt"
  local pred_out_root="${PRED_ROOT}/${exp_name}"

  python scripts/train_sam.py \
    --dataset "${DATASET}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --epochs 20 \
    --lr 5e-6 \
    --num-workers 16 \
    --max-instances 64 \
    --box-padding-ratio 0.05 \
    --box-jitter-ratio 0.08 \
    --cache-embeddings \
    --output "${finetuned_checkpoint}" \
    ${train_extra_args}

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --finetuned-checkpoint "${finetuned_checkpoint}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --save-overlays \
    --out-root "${pred_out_root}" \
    ${infer_extra_args}

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --pred-dir "${pred_out_root}/${DATASET}/${SPLIT}/masks"
}

run_checkpoint_eval() {
  local exp_name="$1"
  local finetuned_checkpoint="$2"
  local infer_extra_args="${3:-}"
  local pred_out_root="${PRED_ROOT}/${exp_name}"

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --finetuned-checkpoint "${finetuned_checkpoint}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --save-overlays \
    --out-root "${pred_out_root}" \
    ${infer_extra_args}

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --pred-dir "${pred_out_root}/${DATASET}/${SPLIT}/masks"
}

python scripts/download_sam_checkpoint.py
python scripts/download_medsam_checkpoint.py

# 0. Zero-shot baselines.
run_zero_shot "zero_shot_box_only_conf020_pad005"
run_medsam_zero_shot "medsam_zero_shot_box_only_conf020_pad005"
run_zero_shot \
  "zero_shot_dynamic_box_consistency_conf020" \
  "--prompt-mode box --dynamic-correction --dynamic-padding-ratios 0.00,0.03,0.05,0.08,0.10,0.15 --dynamic-base-padding-ratio 0.05"
run_zero_shot \
  "zero_shot_mask_to_prompt_refine_conf020_pad005" \
  "--prompt-mode box --mask-to-prompt-refine --refine-box-padding-ratio 0.03 --refine-num-positive-points 1 --refine-num-negative-points 4 --refine-negative-dilate-kernel 9"
run_zero_shot \
  "zero_shot_box_neg_ring_conf020_pad005" \
  "--prompt-mode box+neg --num-negative-points 8 --negative-point-offset-ratio 0.08"
run_zero_shot \
  "zero_shot_box_fgbg_ring_conf020_pad005" \
  "--prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08"

# 1. Conservative mask-decoder fine-tuning with box only.
run_finetune_ablation "maskdec_boxonly_e20_lr5e-6_pad005_jit008" ""

# 2. Fine-tuning with structured background negative points.
run_finetune_ablation \
  "maskdec_boxneg_heat_contrast_e20_lr5e-6_pad005_jit008" \
  "--prompt-mode box+neg --prompt-loss-weight 0.25 --prompt-heatmap-loss-weight 0.15 --contrastive-loss-weight 0.05" \
  "--prompt-mode box+neg --num-negative-points 8 --negative-point-offset-ratio 0.08"

# 3. Fine-tuning with structured foreground/background points and prompt encoder learning.
run_finetune_ablation \
  "maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008" \
  "--prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08 --train-prompt-encoder --prompt-loss-weight 0.25 --prompt-heatmap-loss-weight 0.15 --contrastive-loss-weight 0.05" \
  "--prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08"

# 4. Same checkpoint, but evaluate with GT boxes to diagnose YOLO box mismatch.
run_checkpoint_eval \
  "maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008_gtbox" \
  "${FINE_TUNE_ROOT}/maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008.pt" \
  "--box-source gt --prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08"

# 5. Add GBC adapter on top of prompt learning.
run_finetune_ablation \
  "maskdec_prompt_gbc_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008" \
  "--prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08 --train-prompt-encoder --prompt-loss-weight 0.25 --prompt-heatmap-loss-weight 0.15 --contrastive-loss-weight 0.05 --use-gbc" \
  "--prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08"
