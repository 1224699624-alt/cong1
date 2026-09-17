#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"; PY="/root/miniconda3/bin/python"; BIN="/root/miniconda3/bin"; cd "$ROOT"; export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONHASHSEED=261 nnUNet_compile=false
RUN="outputs/nnunet/r261_conditional_seam"; DATA="$RUN/data"; export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"; LOG="outputs/bridge_logs/r261_conditional_seam_nnunet.log"; mkdir -p outputs/bridge_logs "$RUN"; exec > >(tee -a "$LOG") 2>&1; echo "[R261] start $(date -Is)"
$PY - <<'PY'
import importlib.metadata
assert importlib.metadata.version('nnunetv2')=='2.8.1'
PY
joined="data/raw/TSRS_RSNA-Epiphysis data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1 $RUN"
[[ "${joined,,}" != *clean-test* && "${joined,,}" != *articular* ]]
MATURE="outputs/nnunet/r202_mature_import/checkpoint_best.pth"; test "$(sha256sum "$MATURE"|cut -d' ' -f1)" = "65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7"
$PY -m py_compile scripts/build_r261_conditional_seam_priors.py scripts/prepare_r261_nnunet_dataset.py scripts/patch_r261_nnunet_plans.py scripts/adapt_r202_checkpoint_r261.py scripts/prepare_r261_val_inputs.py scripts/nnunet_trainers/nnUNetTrainerR261ConditionalSeam.py
$PY scripts/build_r261_conditional_seam_priors.py --overwrite
$PY scripts/prepare_r261_nnunet_dataset.py --overwrite
nnUNetv2_extract_fingerprint -d 205 --verify_dataset_integrity
nnUNetv2_plan_experiment -d 205
$PY scripts/patch_r261_nnunet_plans.py --plans "$nnUNet_preprocessed/Dataset205_TSRS_RSNAEpiphysisConditionalSeam2D/nnUNetPlans.json"
nnUNetv2_preprocess -d 205 -c 2d
cp "$DATA/splits_final_r261.json" "$nnUNet_preprocessed/Dataset205_TSRS_RSNAEpiphysisConditionalSeam2D/splits_final.json"
$PY scripts/adapt_r202_checkpoint_r261.py --input "$MATURE" --output "$RUN/r202_thirteen_channel_zero_prior.pth" | tee "$RUN/adaptation_manifest.json"; export R261_ADAPTED_CHECKPOINT="$RUN/r202_thirteen_channel_zero_prior.pth"
TRAINER_DST="$($PY - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss/nnUNetTrainerR261ConditionalSeam.py')
PY
)"; cp scripts/nnunet_trainers/nnUNetTrainerR261ConditionalSeam.py "$TRAINER_DST"; $PY -m py_compile "$TRAINER_DST"
SANITY="$nnUNet_results/Dataset205_TSRS_RSNAEpiphysisConditionalSeam2D/nnUNetTrainerR261ConditionalSeamSanity__nnUNetPlans__2d/fold_0"; rm -rf "${SANITY%/fold_0}"; nnUNetv2_train 205 2d 0 -tr nnUNetTrainerR261ConditionalSeamSanity
$PY - "$SANITY/r261_training_dynamics.jsonl" <<'PY'
import json,math,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
assert len(rows)==2 and all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(int,float)))
assert max(r['valid_slots'] for r in rows)>0 and max(r['alpha'] for r in rows)>0
print({'sanity':'passed','rows':rows})
PY
OUT="$nnUNet_results/Dataset205_TSRS_RSNAEpiphysisConditionalSeam2D/nnUNetTrainerR261ConditionalSeam__nnUNetPlans__2d/fold_0"; rm -rf "${OUT%/fold_0}"; nnUNetv2_train 205 2d 0 -tr nnUNetTrainerR261ConditionalSeam
test -s "$OUT/checkpoint_best.pth"; test -s "$OUT/r261_early_stop.json"
$PY scripts/prepare_r261_val_inputs.py --output "$RUN/imagesValPrior"; rm -rf "$RUN/predictions"; nnUNetv2_predict -i "$RUN/imagesValPrior" -o "$RUN/predictions" -d 205 -c 2d -f 0 -tr nnUNetTrainerR261ConditionalSeam -chk checkpoint_best.pth
ACTIVE="outputs/ablations/r261_conditional_seam/TSRS_RSNA-Epiphysis/val/masks"; rm -rf "$ACTIVE"; $PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions" --mask-dir "$ACTIVE" --expected-count 96 --strip-prefix val_
BASE="outputs/ablations/r202_mature_baseline/TSRS_RSNA-Epiphysis/val/masks"; $PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$BASE" --output outputs/analysis/r261_mature_baseline_original_val_r201.json --expected-count 96; $PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$ACTIVE" --output outputs/analysis/r261_conditional_seam_original_val_r201.json --expected-count 96
VIS="outputs/visualizations/r261_conditional_seam_original_val"; rm -rf "$VIS"; $PY scripts/render_r259_nnunet_prior_comparison.py --image-dir data/raw/TSRS_RSNA-Epiphysis/val --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --baseline-dir "$BASE" --improved-dir "$ACTIVE" --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv --output-dir "$VIS" --num-cases 12 --baseline-label "Mature R202 nnU-Net" --improved-label "R261 conditional seam reasoning"
echo "[R261] done $(date -Is)"
