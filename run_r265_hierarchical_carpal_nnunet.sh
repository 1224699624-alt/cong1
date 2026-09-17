#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/YOLO_SAM_generic_src

ROOT="$PWD/outputs/nnunet/r265_hierarchical_carpal"
DATA="$ROOT/data"
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$ROOT/nnUNet_results"
export nnUNet_compile=false
export R265_ADAPTED_CHECKPOINT="$ROOT/r202_thirteen_channel_zero_prior.pth"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT" outputs/bridge_logs

echo "[R265] prepare dataset $(date -Iseconds)"
/root/miniconda3/bin/python scripts/prepare_r265_nnunet_dataset.py --overwrite
nnUNetv2_extract_fingerprint -d 206 --verify_dataset_integrity
nnUNetv2_plan_experiment -d 206
/root/miniconda3/bin/python scripts/patch_r261_nnunet_plans.py --plans "$nnUNet_preprocessed/Dataset206_TSRS_RSNAEpiphysisHierarchicalCarpal2D/nnUNetPlans.json"
nnUNetv2_preprocess -d 206 -c 2d -np 8
cp "$DATA/splits_final_r265.json" "$nnUNet_preprocessed/Dataset206_TSRS_RSNAEpiphysisHierarchicalCarpal2D/splits_final.json"

/root/miniconda3/bin/python scripts/adapt_r202_checkpoint_r261.py \
  --input outputs/nnunet/r202_mature_import/checkpoint_best.pth \
  --output "$R265_ADAPTED_CHECKPOINT"

TRAINER_DIR=$(/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__path__[0])/'training/nnUNetTrainer/variants/loss')
PY
)
cp scripts/nnunet_trainers/nnUNetTrainerR261ConditionalSeam.py "$TRAINER_DIR/"
cp scripts/nnunet_trainers/nnUNetTrainerR265HierarchicalCarpal.py "$TRAINER_DIR/"

echo "[R265] two-epoch GPU sanity $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 206 2d 0 -tr nnUNetTrainerR265HierarchicalCarpalSanity
SANITY="$nnUNet_results/Dataset206_TSRS_RSNAEpiphysisHierarchicalCarpal2D/nnUNetTrainerR265HierarchicalCarpalSanity__nnUNetPlans__2d/fold_0/r265_training_dynamics.jsonl"
/root/miniconda3/bin/python - "$SANITY" <<'PY'
import json,math,sys
rows=[json.loads(x) for x in open(sys.argv[1],encoding='utf-8') if x.strip()]
assert len(rows)>=2, len(rows)
for r in rows:
 for k in ('relation_loss','alpha','g_base','g_relation','valid_slots','gap_probability','support_probability'):
  assert math.isfinite(float(r[k])), (k,r)
 assert float(r['valid_slots'])>0, r
assert any(float(r['relation_loss'])>0 for r in rows), rows
print(json.dumps({'sanity_passed':True,'rows':rows},indent=2))
PY

echo "[R265] mature training $(date -Iseconds)"
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 206 2d 0 -tr nnUNetTrainerR265HierarchicalCarpal
echo "[R265] training complete $(date -Iseconds)"
