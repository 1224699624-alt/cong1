#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

RUN_ID="r202_nnunet2d"
DATASET_ID=202
DATASET_NAME="TSRS_RSNAEpiphysis2D"
DATASET="Dataset${DATASET_ID}_${DATASET_NAME}"
NNUNET_ROOT="$PWD/outputs/nnunet/r202"
LOG_DIR="$PWD/outputs/bridge_logs"
LOG_FILE="$LOG_DIR/${RUN_ID}.log"
PRED_DIR="$NNUNET_ROOT/predictions/$DATASET"
MASK_DIR="$PWD/outputs/ablations_variants/${RUN_ID}/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks"
METRICS_JSON="$PWD/outputs/analysis/${RUN_ID}_r201_unified_metrics.json"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[R202] start $(date -Is)"
echo "[R202] cwd=$PWD"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
else
  echo "[R202] conda not found" >&2
  exit 1
fi

ENV_NAME="${R202_ENV_NAME:-aris-nnunet-r202-fast}"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" --clone yolo-sam-gpu
fi
conda activate "$ENV_NAME"

python -m pip install -U pip
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
python -m pip install -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" --timeout 120 --retries 8 --no-deps "nnunetv2==2.8.1"
python -m pip install -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" --timeout 120 --retries 8 "acvl-utils>=0.2.6,<0.3" "dynamic-network-architectures>=0.4.4,<0.5" "batchgenerators>=0.25.1" "batchgeneratorsv2>=0.3.2" SimpleITK nibabel yacs graphviz blosc2 connected-components-3d seaborn
python - <<'PY'
import torch
import nnunetv2
print("[R202] torch", torch.__version__, "cuda", torch.cuda.is_available())
print("[R202] nnunetv2", nnunetv2.__file__)
PY

export nnUNet_raw="$NNUNET_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_ROOT/nnUNet_preprocessed"
export nnUNet_results="$NNUNET_ROOT/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

python scripts/prepare_r202_nnunet_dataset.py \
  --raw-root data/raw \
  --dataset TSRS_RSNA-Epiphysis \
  --clean-label-root data/raw_variants/TSRS_RSNA-Epiphysis_clean_test_v2/test_labels \
  --nnunet-root "$NNUNET_ROOT" \
  --dataset-id "$DATASET_ID" \
  --dataset-name "$DATASET_NAME" \
  --overwrite

nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

SPLIT_SRC="$NNUNET_ROOT/splits_final_r202.json"
SPLIT_DST="$nnUNet_preprocessed/$DATASET/splits_final.json"
cp "$SPLIT_SRC" "$SPLIT_DST"
echo "[R202] fixed train/val split copied to $SPLIT_DST"

nnUNetv2_train "$DATASET_ID" 2d 0

rm -rf "$PRED_DIR"
mkdir -p "$PRED_DIR"
nnUNetv2_predict \
  -i "$nnUNet_raw/$DATASET/imagesTs" \
  -o "$PRED_DIR" \
  -d "$DATASET_ID" \
  -c 2d \
  -f 0

python scripts/convert_r202_nnunet_predictions.py \
  --pred-dir "$PRED_DIR" \
  --mask-dir "$MASK_DIR" \
  --summary outputs/analysis/r202_nnunet2d_prediction_conversion_summary.json \
  --expected-count 81

python scripts/evaluate_masks.py \
  --dataset TSRS_RSNA-Epiphysis_clean_test_v2 \
  --split test \
  --raw-root data/raw_variants \
  --pred-dir "$MASK_DIR" \
  --output outputs/analysis/r202_nnunet2d_clean_test_v2_metrics.json

python scripts/run_r201_unified_eval.py

cp outputs/analysis/r201_unified_eval/r202_nnunet2d_unified_metrics.json "$METRICS_JSON" || true

echo "[R202] done $(date -Is)"
