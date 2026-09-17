#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"; PY="/root/miniconda3/bin/python"; BIN="/root/miniconda3/bin"; cd "$ROOT"; export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONHASHSEED=259
RUN="outputs/nnunet/r259_frozen_prior"; DATA="$RUN/data"; export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"
LOG="outputs/bridge_logs/r259_nnunet_frozen_prior_visual_probe.log"; mkdir -p outputs/bridge_logs outputs/analysis "$RUN"; exec > >(tee -a "$LOG") 2>&1
echo "[R259] start $(date -Is)"
CHECKPOINT_DIR="outputs/pair_prior/r258b_prediction_relation_prior_full_20260715_133745"
test "$(find "$CHECKPOINT_DIR" -maxdepth 1 -name 'image_centers_age_sex_seed*_final.pt' | wc -l)" -eq 3
$PY -m py_compile scripts/build_r259_frozen_prior_maps.py scripts/prepare_r259_nnunet_prior_dataset.py scripts/prepare_r259_val_inputs.py scripts/nnunet_trainers/nnUNetTrainerR259FrozenPrior.py scripts/render_r259_nnunet_prior_comparison.py
$PY scripts/build_r259_frozen_prior_maps.py --checkpoint-dir "$CHECKPOINT_DIR" --output-root outputs/priors/r259_frozen_relation --device cuda
$PY scripts/prepare_r259_nnunet_prior_dataset.py --prior-root outputs/priors/r259_frozen_relation --nnunet-root "$DATA" --overwrite
nnUNetv2_plan_and_preprocess -d 203 -c 2d --verify_dataset_integrity
cp "$DATA/splits_final_r259.json" "$nnUNet_preprocessed/Dataset203_TSRS_RSNAEpiphysisPrior2D/splits_final.json"
TRAINER_DST="$($PY - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss/nnUNetTrainerR259FrozenPrior.py')
PY
)"
mkdir -p "$(dirname "$TRAINER_DST")"; cp scripts/nnunet_trainers/nnUNetTrainerR259FrozenPrior.py "$TRAINER_DST"; $PY -m py_compile "$TRAINER_DST"
$PY scripts/prepare_r259_val_inputs.py --native-dir "$RUN/imagesValNative" --active-dir "$RUN/imagesValPrior"
DS="Dataset203_TSRS_RSNAEpiphysisPrior2D"; W="$nnUNet_results/$DS/nnUNetTrainerR259Warmup__nnUNetPlans__2d/fold_0"; N="$nnUNet_results/$DS/nnUNetTrainerR259Native__nnUNetPlans__2d/fold_0"; A="$nnUNet_results/$DS/nnUNetTrainerR259FrozenPrior__nnUNetPlans__2d/fold_0"
rm -rf "${W%/fold_0}" "${N%/fold_0}" "${A%/fold_0}"
nnUNetv2_train 203 2d 0 -tr nnUNetTrainerR259Warmup
test -s "$W/checkpoint_final.pth"; mkdir -p "$N" "$A"; cp "$W/checkpoint_final.pth" "$N/checkpoint_latest.pth"; cp "$W/checkpoint_final.pth" "$A/checkpoint_latest.pth"
SHA="$(sha256sum "$W/checkpoint_final.pth"|cut -d' ' -f1)"; test "$SHA" = "$(sha256sum "$N/checkpoint_latest.pth"|cut -d' ' -f1)"; test "$SHA" = "$(sha256sum "$A/checkpoint_latest.pth"|cut -d' ' -f1)"
$PY - "$W/checkpoint_final.pth" "$RUN/fork_manifest.json" <<'PY'
import hashlib,json,sys,torch
p,out=sys.argv[1:]; c=torch.load(p,map_location='cpu'); assert c['current_epoch']==2 and c['optimizer_state'] and c['logging'] and c['grad_scaler_state'] is not None; json.dump({'checkpoint':p,'sha256':hashlib.sha256(open(p,'rb').read()).hexdigest(),'current_epoch':2,'optimizer':True,'logging':True,'grad_scaler':True},open(out,'w'),indent=2)
PY
nnUNetv2_train 203 2d 0 -tr nnUNetTrainerR259Native --c
nnUNetv2_train 203 2d 0 -tr nnUNetTrainerR259FrozenPrior --c
test -s "$A/r259_prior_dynamics.jsonl"
rm -rf "$RUN/predictions_native" "$RUN/predictions_prior"
nnUNetv2_predict -i "$RUN/imagesValNative" -o "$RUN/predictions_native" -d 203 -c 2d -f 0 -tr nnUNetTrainerR259Native -chk checkpoint_final.pth
nnUNetv2_predict -i "$RUN/imagesValPrior" -o "$RUN/predictions_prior" -d 203 -c 2d -f 0 -tr nnUNetTrainerR259FrozenPrior -chk checkpoint_final.pth
NATIVE="outputs/ablations/r259_native/TSRS_RSNA-Epiphysis/val/masks"; ACTIVE="outputs/ablations/r259_frozen_prior/TSRS_RSNA-Epiphysis/val/masks"; rm -rf "$NATIVE" "$ACTIVE"
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions_native" --mask-dir "$NATIVE" --expected-count 96 --strip-prefix val_
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions_prior" --mask-dir "$ACTIVE" --expected-count 96 --strip-prefix val_
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1/val_labels --pred-dir "$NATIVE" --output outputs/analysis/r259_native_original_val_r201.json --expected-count 96
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1/val_labels --pred-dir "$ACTIVE" --output outputs/analysis/r259_frozen_prior_original_val_r201.json --expected-count 96
$PY scripts/render_r259_nnunet_prior_comparison.py --image-dir data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1/val --gt-dir data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1/val_labels --baseline-dir "$NATIVE" --improved-dir "$ACTIVE" --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv --output-dir outputs/visualizations/r259_nnunet_frozen_prior_original_val --num-cases 12 --xray-label "Contrast-normalized X-ray"
$PY - <<'PY'
import json
from pathlib import Path
p=Path('outputs/visualizations/r259_nnunet_frozen_prior_original_val'); m=json.loads((p/'manifest.json').read_text()); assert len(m)==12 and len(list(p.glob('*_comparison.png')))==12
PY
echo "[R259] done $(date -Is)"
