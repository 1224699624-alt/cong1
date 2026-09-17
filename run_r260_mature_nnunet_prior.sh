#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"; PY="/root/miniconda3/bin/python"; BIN="/root/miniconda3/bin"; cd "$ROOT"; export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONHASHSEED=260 nnUNet_compile=false
RUN="outputs/nnunet/r260_mature_prior"; DATA="$RUN/data"; export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"; LOG="outputs/bridge_logs/r260_mature_nnunet_prior.log"; mkdir -p outputs/bridge_logs "$RUN"; exec > >(tee -a "$LOG") 2>&1; echo "[R260] start $(date -Is)"
MATURE="outputs/nnunet/r202_mature_import/checkpoint_best.pth"; BASE="outputs/ablations/r202_mature_baseline/TSRS_RSNA-Epiphysis/val/masks"; test "$(sha256sum "$MATURE"|cut -d' ' -f1)" = "65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7"
$PY - <<'PY'
import hashlib, importlib.metadata, json
from pathlib import Path
if importlib.metadata.version('nnunetv2') != '2.8.1': raise RuntimeError('R260 requires nnunetv2==2.8.1')
base=Path('outputs/ablations/r202_mature_baseline/TSRS_RSNA-Epiphysis/val/masks')
gt=Path('data/raw/TSRS_RSNA-Epiphysis/val_labels')
files=sorted(base.glob('*.png'))
if len(files)!=96 or [p.name for p in files]!=sorted(p.name for p in gt.glob('*.png')): raise RuntimeError('R202 baseline names do not exactly match 96 original-val GT names')
h=hashlib.sha256()
for p in files: h.update(p.name.encode()); h.update(b'\0'); h.update(p.read_bytes())
digest=h.hexdigest()
if digest!='2f685e548cdbc853814eba7cd351b351210bc7411d3ce71a0501b90bfb3a0457': raise RuntimeError(f'R202 baseline mask hash mismatch: {digest}')
manifest={'source_checkpoint_sha256':'65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7','mask_count':96,'mask_directory_sha256':digest,'exact_original_val_names':True,'nnunetv2':'2.8.1'}
Path('outputs/nnunet/r260_mature_prior').mkdir(parents=True,exist_ok=True)
Path('outputs/nnunet/r260_mature_prior/r202_baseline_provenance.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
PY
$PY -m py_compile scripts/prepare_r260_mature_prior_dataset.py scripts/adapt_r202_checkpoint_two_channel.py scripts/prepare_r260_val_inputs.py scripts/nnunet_trainers/nnUNetTrainerR260MaturePrior.py scripts/render_r259_nnunet_prior_comparison.py
if [[ "${R260_REUSE_PREPROCESSED:-0}" != "1" ]]; then
  $PY scripts/prepare_r260_mature_prior_dataset.py --overwrite; nnUNetv2_plan_and_preprocess -d 204 -c 2d --verify_dataset_integrity; cp "$DATA/splits_final_r260.json" "$nnUNet_preprocessed/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/splits_final.json"
else
  test -s "$nnUNet_preprocessed/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetPlans.json"; test -s "$nnUNet_preprocessed/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/splits_final.json"; echo "[R260] reusing verified preprocessed Dataset204"
fi
$PY scripts/adapt_r202_checkpoint_two_channel.py --input "$MATURE" --output "$RUN/r202_two_channel_zero_prior.pth" | tee "$RUN/adaptation_manifest.json"; export R260_ADAPTED_CHECKPOINT="$RUN/r202_two_channel_zero_prior.pth"
TRAINER_DST="$($PY - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss/nnUNetTrainerR260MaturePrior.py')
PY
)"; cp scripts/nnunet_trainers/nnUNetTrainerR260MaturePrior.py "$TRAINER_DST"; $PY -m py_compile "$TRAINER_DST"; $PY scripts/prepare_r260_val_inputs.py --output "$RUN/imagesValPrior"
OUT="$nnUNet_results/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetTrainerR260MaturePrior__nnUNetPlans__2d/fold_0"; rm -rf "${OUT%/fold_0}"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR260MaturePrior
test -s "$OUT/checkpoint_best.pth"; test -s "$OUT/r260_early_stop.json"; rm -rf "$RUN/predictions_prior"; nnUNetv2_predict -i "$RUN/imagesValPrior" -o "$RUN/predictions_prior" -d 204 -c 2d -f 0 -tr nnUNetTrainerR260MaturePrior -chk checkpoint_best.pth
ACTIVE="outputs/ablations/r260_mature_prior/TSRS_RSNA-Epiphysis/val/masks"; rm -rf "$ACTIVE"; $PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions_prior" --mask-dir "$ACTIVE" --expected-count 96 --strip-prefix val_
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$BASE" --output outputs/analysis/r260_mature_baseline_original_val_r201.json --expected-count 96; $PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$ACTIVE" --output outputs/analysis/r260_mature_prior_original_val_r201.json --expected-count 96
VIS="outputs/visualizations/r260_mature_nnunet_prior_original_val"; rm -rf "$VIS"; $PY scripts/render_r259_nnunet_prior_comparison.py --image-dir data/raw/TSRS_RSNA-Epiphysis/val --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --baseline-dir "$BASE" --improved-dir "$ACTIVE" --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv --output-dir "$VIS" --num-cases 12 --xray-label "Original X-ray" --baseline-label "Mature R202 nnU-Net" --improved-label "Mature + frozen prior/loss"
$PY - <<'PY'
import json
from pathlib import Path
p=Path('outputs/visualizations/r260_mature_nnunet_prior_original_val'); assert len(json.loads((p/'manifest.json').read_text()))==12 and len(list(p.glob('*_comparison.png')))==12
PY
echo "[R260] done $(date -Is)"
