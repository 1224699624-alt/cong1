#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${R255_PROJECT_ROOT_OVERRIDE:-/home/shenzeyu/workspace/YOLO_SAM_generic_src}"
cd "$PROJECT_ROOT"

RUN_ID="r255_close_gap_nnunet_sanity"
PY="${R255_PYTHON:-/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin/python}"
BIN="${R255_BIN:-$(dirname "$PY")}"
GPU="${R255_GPU:-1}"
R202_ROOT="$PWD/outputs/nnunet/r202"
ROOT="$PWD/outputs/nnunet/r255_close_gap"
RESULTS="$ROOT/nnUNet_results"
DATASET="Dataset202_TSRS_RSNAEpiphysis2D"
BASELINE_MODEL="$R202_ROOT/nnUNet_results/$DATASET/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
TRAIN_METADATA="$PWD/outputs/metadata/r255/filtered_train.csv"
VAL_METADATA="$PWD/outputs/metadata/r255/filtered_val.csv"
GATE_JSON="$PWD/outputs/analysis/r255_gate_a/r255_close_gap_gate_a.json"
GATE_CSV="$PWD/outputs/analysis/r255_gate_a/r255_close_gap_pairs.csv"
LOG="$PWD/outputs/bridge_logs/${RUN_ID}.log"

mkdir -p "$PWD/outputs/bridge_logs" "$ROOT" "$RESULTS" "$PWD/outputs/analysis"
exec > >(tee -a "$LOG") 2>&1
echo "[R255] start $(date -Is)"
for f in "$PY" "$BASELINE_MODEL" "$TRAIN_METADATA" "$VAL_METADATA" "$GATE_JSON" "$GATE_CSV"; do test -e "$f"; done
$PY - "$TRAIN_METADATA" "$VAL_METADATA" "$GATE_JSON" "$GATE_CSV" <<'PY'
import csv, hashlib, json, sys
train,val,gate_json,gate_csv=sys.argv[1:]
sha=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
assert sha(train)=='bfb40b05d22a6b2151af3982b9de2e0dd2534b21c661df79037f33247d599857'
assert sha(val)=='d52cb2e32f055c315e47fe3e010ac7e55fbfc2018fe9419ff0da42848d706128'
gate=json.load(open(gate_json)); assert gate['gate_pass'] and not gate['clean_test_used']
rows=list(csv.DictReader(open(gate_csv,encoding='utf-8')))
assert not any(r['instance_i']==r['instance_j'] and 1 <= float(r['gap_px']) < 5 for r in rows)
print('[R255] validated input hashes and Gate A')
PY

export PATH="$BIN:$PATH"
export nnUNet_raw="$R202_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$R202_ROOT/nnUNet_preprocessed"
export nnUNet_results="$RESULTS"
export R255_PROJECT_ROOT="$PWD"
export R255_METADATA_CSV="$TRAIN_METADATA"
export R255_GATE_CSV="$GATE_CSV"
export R255_CALIBRATION_STEPS=16
export R255_GMAX=0.25
export PYTHONHASHSEED=255

TRAINER_DST="$($PY - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss/nnUNetTrainerR255CloseGap.py')
PY
)"
mkdir -p "$(dirname "$TRAINER_DST")"
cp scripts/nnunet_trainers/nnUNetTrainerR255CloseGap.py "$TRAINER_DST"
$PY -m py_compile "$TRAINER_DST" scripts/evaluate_r255_close_gap.py scripts/summarize_r255_gate_b.py
$PY scripts/prepare_r254_nnunet_val_input.py --output "$ROOT/imagesVal"

WARMUP_FOLDER="$RESULTS/$DATASET/nnUNetTrainerR255Warmup__nnUNetPlans__2d/fold_0"
NATIVE_FOLDER="$RESULTS/$DATASET/nnUNetTrainerR255NativeSanity__nnUNetPlans__2d/fold_0"
ACTIVE_FOLDER="$RESULTS/$DATASET/nnUNetTrainerR255CloseGapSanity__nnUNetPlans__2d/fold_0"
rm -rf "${WARMUP_FOLDER%/fold_0}" "${NATIVE_FOLDER%/fold_0}" "${ACTIVE_FOLDER%/fold_0}"

# Epochs 0-1 use a shared four-epoch PolyLR horizon.
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train 202 2d 0 -tr nnUNetTrainerR255Warmup -pretrained_weights "$BASELINE_MODEL"
FORK_CHECKPOINT="$WARMUP_FOLDER/checkpoint_final.pth"
test -s "$FORK_CHECKPOINT"
mkdir -p "$NATIVE_FOLDER" "$ACTIVE_FOLDER"
cp "$FORK_CHECKPOINT" "$NATIVE_FOLDER/checkpoint_latest.pth"
cp "$FORK_CHECKPOINT" "$ACTIVE_FOLDER/checkpoint_latest.pth"
FORK_SHA="$(sha256sum "$FORK_CHECKPOINT" | cut -d' ' -f1)"
test "$FORK_SHA" = "$(sha256sum "$NATIVE_FOLDER/checkpoint_latest.pth" | cut -d' ' -f1)"
test "$FORK_SHA" = "$(sha256sum "$ACTIVE_FOLDER/checkpoint_latest.pth" | cut -d' ' -f1)"
$PY - "$FORK_CHECKPOINT" "$FORK_SHA" "$ROOT/fork_manifest.json" <<'PY'
import json,sys,torch
p,sha,out=sys.argv[1:]; c=torch.load(p,map_location='cpu',weights_only=False)
assert c['current_epoch']==2, c['current_epoch']
json.dump({'checkpoint':p,'sha256':sha,'current_epoch':2,'optimizer_state_present':bool(c['optimizer_state']),'grad_scaler_state_present':c['grad_scaler_state'] is not None},open(out,'w'),indent=2)
PY

# Epochs 2-3 resume the identical full network/optimizer/AMP/logger state.
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train 202 2d 0 -tr nnUNetTrainerR255NativeSanity --c
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_train 202 2d 0 -tr nnUNetTrainerR255CloseGapSanity --c
DYNAMICS="$ACTIVE_FOLDER/r255_training_dynamics.jsonl"
test -s "$DYNAMICS"
$PY - "$DYNAMICS" <<'PY'
import json,math,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]; assert len(rows)==2, rows
for row in rows:
    for key in ('base_loss','pair_loss','valid_pairs','anchors','anchor_probability','anchor_gt_foreground_fraction'):
        assert math.isfinite(float(row[key])), (key,row)
assert max(r['valid_pairs'] for r in rows)>0 and max(r['anchors'] for r in rows)>0
assert max(r['anchor_gt_foreground_fraction'] for r in rows)==0
assert rows[-1]['alpha_frozen'] is not None
assert max(float(r['alpha']) for r in rows)>0, 'close loss never became active after calibration'
print('[R255] training sanity passed',rows[-1])
PY

NATIVE_PRED="$ROOT/predictions/native_original_val"
ACTIVE_PRED="$ROOT/predictions/r255_original_val"
rm -rf "$NATIVE_PRED" "$ACTIVE_PRED"
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_predict -i "$ROOT/imagesVal" -o "$NATIVE_PRED" -d 202 -c 2d -f 0 -tr nnUNetTrainerR255NativeSanity -chk checkpoint_final.pth
CUDA_VISIBLE_DEVICES="$GPU" nnUNetv2_predict -i "$ROOT/imagesVal" -o "$ACTIVE_PRED" -d 202 -c 2d -f 0 -tr nnUNetTrainerR255CloseGapSanity -chk checkpoint_final.pth

NATIVE_MASKS="$PWD/outputs/ablations/r255_native_sanity/TSRS_RSNA-Epiphysis/val/masks"
ACTIVE_MASKS="$PWD/outputs/ablations/r255_close_gap_sanity/TSRS_RSNA-Epiphysis/val/masks"
rm -rf "$NATIVE_MASKS" "$ACTIVE_MASKS"
$PY scripts/convert_nnunet_predictions.py --pred-dir "$NATIVE_PRED" --mask-dir "$NATIVE_MASKS" --expected-count 96
$PY scripts/convert_nnunet_predictions.py --pred-dir "$ACTIVE_PRED" --mask-dir "$ACTIVE_MASKS" --expected-count 96

NATIVE_R201="$PWD/outputs/analysis/r255_native_sanity_original_val_r201.json"
ACTIVE_R201="$PWD/outputs/analysis/r255_close_gap_sanity_original_val_r201.json"
CLOSE_JSON="$PWD/outputs/analysis/r255_close_gap_sanity_original_val_diagnostics.json"
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$NATIVE_MASKS" --output "$NATIVE_R201" --expected-count 96
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$ACTIVE_MASKS" --output "$ACTIVE_R201" --expected-count 96
$PY scripts/evaluate_r255_close_gap.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --baseline-dir "$NATIVE_MASKS" --r255-dir "$ACTIVE_MASKS" --metadata-csv "$VAL_METADATA" --gate-json "$GATE_JSON" --output-json "$CLOSE_JSON" --output-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv --expected-count 96
$PY scripts/summarize_r255_gate_b.py --native-r201 "$NATIVE_R201" --active-r201 "$ACTIVE_R201" --close-diagnostics "$CLOSE_JSON" --output "$ROOT/gate_b_decision.json"
echo "[R255] done $(date -Is)"
