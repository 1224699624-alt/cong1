#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
OUT=/root/final_prior_v1/ram
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1

MODE="${1:-audit}"
PYTHON="/root/miniconda3/bin/python"
SCRIPT="scripts/train_final_prior_ram.py"
COMMON=(
  --dataset-root "$ROOT/data/remote_variants/RAM-W600"
  --prior-root "$ROOT/outputs/priors/r323_ram_native_r317_single_seed"
  --native-checkpoint "$ROOT/outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth"
  --r332-checkpoint "$ROOT/outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth"
  --overlap-checkpoint /root/r351_workspace/overlap/full/best.pth
  --epochs 2 --iterations-per-epoch 100 --seed 3511
)

case "$MODE" in
  audit)
    "$PYTHON" "$SCRIPT" "${COMMON[@]}" --variant D --training-mode adapter_only \
      --audit --output "$OUT/audit"
    ;;
  baseline|baseline-only)
    "$PYTHON" "$SCRIPT" "${COMMON[@]}" --baseline-only --output "$OUT/baseline"
    ;;
  R0|A|a)
    "$PYTHON" "$SCRIPT" "${COMMON[@]}" --variant A --training-mode joint_finetune \
      --output "$OUT/R0"
    ;;
  B|b)
    "$PYTHON" "$SCRIPT" "${COMMON[@]}" --variant B --training-mode joint_finetune \
      --output "$OUT/B"
    ;;
  R1|C|c)
    "$PYTHON" "$SCRIPT" "${COMMON[@]}" --variant C --training-mode joint_finetune \
      --output "$OUT/R1"
    ;;
  *)
    echo "usage: $0 {audit|baseline|R0|R1}" >&2
    exit 2
    ;;
esac
