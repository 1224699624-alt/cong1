#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

RUN_ID="r254_hapdsp_pcr_nnunet"
PY="/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin/python"
BIN="/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin"
R202_ROOT="$PWD/outputs/nnunet/r202"
R254_ROOT="$PWD/outputs/nnunet/r254_hapdsp_pcr"
R254_RESULTS="$R254_ROOT/nnUNet_results"
DATASET="Dataset202_TSRS_RSNAEpiphysis2D"
BASELINE_MODEL="$R202_ROOT/nnUNet_results/$DATASET/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
METADATA="$PWD/outputs/metadata/r254/filtered_train.csv"
LOG="$PWD/outputs/bridge_logs/${RUN_ID}.log"

mkdir -p "$PWD/outputs/bridge_logs" "$R254_ROOT" "$PWD/outputs/analysis" "$PWD/outputs/metadata/r254"
exec > >(tee -a "$LOG") 2>&1

echo "[R254] start $(date -Is)"
test -x "$PY"
test -f "$BASELINE_MODEL"
test -f "$METADATA"
$PY - "$METADATA" <<'PY'
import csv, hashlib, json, sys
path=sys.argv[1]
rows=list(csv.DictReader(open(path, encoding='utf-8-sig')))
assert len(rows) == 875, len(rows)
assert set(rows[0]) == {'id','boneage','male'}, rows[0].keys()
sha=hashlib.sha256(open(path,'rb').read()).hexdigest()
assert sha == 'bfb40b05d22a6b2151af3982b9de2e0dd2534b21c661df79037f33247d599857', sha
print('[R254] metadata', json.dumps({'rows':len(rows),'sha256':sha}))
PY

export PATH="$BIN:$PATH"
export nnUNet_raw="$R202_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$R202_ROOT/nnUNet_preprocessed"
export nnUNet_results="$R254_RESULTS"
export R254_PROJECT_ROOT="$PWD"
export R254_PRIOR_JSON="$PWD/outputs/analysis/r254_hapdsp_train_prior.json"
export R254_METADATA_CSV="$METADATA"
export R254_PCR_RHO="0.05"
export R254_ALPHA_MAX="0.20"
export R254_TOP_FRACTION="0.25"
export R254_WARMUP_EPOCHS="5"
export PYTHONHASHSEED="254"

TRAINER_DST="$($PY - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss/nnUNetTrainerHAPDSPPCR.py')
PY
)"
mkdir -p "$(dirname "$TRAINER_DST")"
cp scripts/nnunet_trainers/nnUNetTrainerHAPDSPPCR.py "$TRAINER_DST"

if $PY - "$R254_PRIOR_JSON" <<'PY'
import json, os, sys
path=sys.argv[1]
assert os.path.exists(path)
p=json.load(open(path))
assert p['num_cases'] == 875
assert p['num_pair_samples'] > 0
assert len(p['feature_names']) == 7
print('[R254] reusing validated train-only prior', path)
PY
then
  :
else
  $PY scripts/build_r254_hapdsp_prior.py \
    --label-dir data/raw/TSRS_RSNA-Epiphysis/train_labels \
    --metadata-csv "$METADATA" \
    --output "$R254_PRIOR_JSON"
fi
$PY scripts/prepare_r254_nnunet_val_input.py --output "$R254_ROOT/imagesVal"

mkdir -p "$R254_RESULTS"
SANITY_FOLDER="$R254_RESULTS/$DATASET/nnUNetTrainerHAPDSPPCRSanity__nnUNetPlans__2d"
rm -rf "$SANITY_FOLDER"
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 202 2d 0 \
  -tr nnUNetTrainerHAPDSPPCRSanity \
  -pretrained_weights "$BASELINE_MODEL"

SANITY_JSONL="$SANITY_FOLDER/fold_0/r254_training_dynamics.jsonl"
test -s "$SANITY_JSONL"
$PY - "$SANITY_JSONL" <<'PY'
import json, math, sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
assert len(rows) == 2, len(rows)
for row in rows:
    for key in ('base_loss','pair_loss','alpha','valid_pairs','anchors','core_probability'):
        assert math.isfinite(float(row[key])), (key,row.get(key))
assert max(float(r['valid_pairs']) for r in rows) > 0, rows
assert max(float(r['anchors']) for r in rows) > 0, rows
assert max(float(r['relation_mean']) for r in rows) >= 0.01, rows
assert max(float(r['k_mean']) for r in rows) >= 0.005, rows
print('[R254] sanity gate passed', rows[-1])
PY

FULL_FOLDER="$R254_RESULTS/$DATASET/nnUNetTrainerHAPDSPPCR__nnUNetPlans__2d"
NATIVE_FOLDER="$R254_RESULTS/$DATASET/nnUNetTrainerR254NativeContinuation__nnUNetPlans__2d"
rm -rf "$FULL_FOLDER" "$NATIVE_FOLDER"
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 202 2d 0 \
  -tr nnUNetTrainerR254NativeContinuation \
  -pretrained_weights "$BASELINE_MODEL"
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 202 2d 0 \
  -tr nnUNetTrainerHAPDSPPCR \
  -pretrained_weights "$BASELINE_MODEL"

BASELINE_PRED="$R254_ROOT/predictions/baseline_original_val"
OURS_PRED="$R254_ROOT/predictions/hapdsp_pcr_original_val"
rm -rf "$BASELINE_PRED" "$OURS_PRED"

export nnUNet_results="$R254_RESULTS"
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
  -i "$R254_ROOT/imagesVal" -o "$BASELINE_PRED" -d 202 -c 2d -f 0 \
  -tr nnUNetTrainerR254NativeContinuation -chk checkpoint_best.pth

export nnUNet_results="$R254_RESULTS"
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict \
  -i "$R254_ROOT/imagesVal" -o "$OURS_PRED" -d 202 -c 2d -f 0 \
  -tr nnUNetTrainerHAPDSPPCR -chk checkpoint_best.pth

BASELINE_MASKS="$PWD/outputs/ablations/r254_nnunet_baseline/TSRS_RSNA-Epiphysis/val/masks"
OURS_MASKS="$PWD/outputs/ablations/r254_hapdsp_pcr_nnunet/TSRS_RSNA-Epiphysis/val/masks"
rm -rf "$BASELINE_MASKS" "$OURS_MASKS"
$PY scripts/convert_nnunet_predictions.py --pred-dir "$BASELINE_PRED" --mask-dir "$BASELINE_MASKS" --expected-count 96
$PY scripts/convert_nnunet_predictions.py --pred-dir "$OURS_PRED" --mask-dir "$OURS_MASKS" --expected-count 96

$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$BASELINE_MASKS" \
  --output outputs/analysis/r254_nnunet_baseline_original_val_r201.json \
  --expected-count 96
$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$OURS_MASKS" \
  --output outputs/analysis/r254_hapdsp_pcr_nnunet_original_val_r201.json \
  --expected-count 96

$PY scripts/render_r254_nnunet_comparison.py \
  --baseline-dir "$BASELINE_MASKS" \
  --ours-dir "$OURS_MASKS" \
  --baseline-metrics outputs/analysis/r254_nnunet_baseline_original_val_r201.json

echo "[R254] done $(date -Is)"
