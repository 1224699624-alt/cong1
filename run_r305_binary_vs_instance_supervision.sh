#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src
PY=/root/miniconda3/bin/python
RUN=outputs/nnunet/r305_binary_vs_instance
DATA=$RUN/data
export nnUNet_raw=$DATA/nnUNet_raw
export nnUNet_preprocessed=$DATA/nnUNet_preprocessed
export nnUNet_results=$RUN/nnUNet_results
export nnUNet_compile=false
export R305_SOURCE_CHECKPOINT=outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/fold_0/checkpoint_best.pth
LOG=outputs/bridge_logs/r305_binary_vs_instance_supervision.log
mkdir -p "$RUN" outputs/bridge_logs
exec > >(tee -a "$LOG") 2>&1
echo "[R305] start $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
$PY -m py_compile scripts/prepare_r305_binary_vs_instance_supervision.py scripts/nnunet_trainers/nnUNetTrainerR305BinaryVsInstance.py scripts/convert_nnunet_predictions.py
$PY scripts/prepare_r305_binary_vs_instance_supervision.py --overwrite
nnUNetv2_plan_and_preprocess -d 305 306 -c 2d --verify_dataset_integrity
cp "$DATA/nnUNet_raw/Dataset305_TSRSBinarySupervision2D/splits_final.json" "$DATA/nnUNet_preprocessed/Dataset305_TSRSBinarySupervision2D/splits_final.json"
cp "$DATA/nnUNet_raw/Dataset306_TSRSInstanceSeamSupervision2D/splits_final.json" "$DATA/nnUNet_preprocessed/Dataset306_TSRSInstanceSeamSupervision2D/splits_final.json"
TRAINER_DST=$($PY - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss/nnUNetTrainerR305BinaryVsInstance.py')
PY
)
cp scripts/nnunet_trainers/nnUNetTrainerR305BinaryVsInstance.py "$TRAINER_DST"
nnUNetv2_train 305 2d 0 -tr nnUNetTrainerR305Binary
nnUNetv2_train 306 2d 0 -tr nnUNetTrainerR305InstanceSeam
rm -rf "$RUN/pred_binary" "$RUN/pred_instance" "$RUN/masks_binary" "$RUN/masks_instance"
nnUNetv2_predict -i "$RUN/imagesVal" -o "$RUN/pred_binary" -d 305 -c 2d -f 0 -tr nnUNetTrainerR305Binary -chk checkpoint_final.pth
nnUNetv2_predict -i "$RUN/imagesVal" -o "$RUN/pred_instance" -d 306 -c 2d -f 0 -tr nnUNetTrainerR305InstanceSeam -chk checkpoint_final.pth
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/pred_binary" --mask-dir "$RUN/masks_binary" --expected-count 96 --strip-prefix val_ --foreground-label 1
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/pred_instance" --mask-dir "$RUN/masks_instance" --expected-count 96 --strip-prefix val_ --foreground-label 1
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$RUN/masks_binary" --output outputs/analysis/r305_binary_supervision_original_val_r201.json --expected-count 96
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$RUN/masks_instance" --output outputs/analysis/r305_instance_supervision_original_val_r201.json --expected-count 96
echo "[R305] complete $(date -Is)"
