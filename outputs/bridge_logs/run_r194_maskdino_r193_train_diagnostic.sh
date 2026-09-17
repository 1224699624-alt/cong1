#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

ENV_PREFIX="/home/shenzeyu/miniconda3/envs/aris-maskdino-r189-cu130"
PY="${ENV_PREFIX}/bin/python"
LOG_DIR="outputs/bridge_logs"
OUT_DIR="outputs/analysis/r194_maskdino_r193_train_diagnostic"
R193_DIR="outputs/analysis/r193_maskdino_pretrained_overfit_gate"
MASKDINO_DIR="external_src/MaskDINO"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="/usr/local/cuda"
export PATH="${ENV_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/shenzeyu/workspace/YOLO_SAM_generic_src/${MASKDINO_DIR}:${PYTHONPATH:-}"

{
  echo "[R194] start $(date -Is)"
  echo "[R194] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

  cat > "${OUT_DIR}/r194_eval_r193_on_train.py" <<'PY'
import json
from pathlib import Path

from detectron2.data.datasets import register_coco_instances

ROOT = Path("/home/shenzeyu/workspace/YOLO_SAM_generic_src")
R193 = ROOT / "outputs/analysis/r193_maskdino_pretrained_overfit_gate"
OUT = ROOT / "outputs/analysis/r194_maskdino_r193_train_diagnostic"
COCO = R193 / "coco"
register_coco_instances("epiphysis_r194_train_eval", {}, str(COCO / "train.json"), str(COCO / "train"))

import sys
sys.path.insert(0, str(ROOT / "external_src/MaskDINO"))
from train_net import main
from detectron2.engine import default_argument_parser

args = default_argument_parser().parse_args([
    "--config-file", str(ROOT / "external_src/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"),
    "--num-gpus", "1",
    "--eval-only",
    "MODEL.WEIGHTS", str(R193 / "maskdino_output/model_final.pth"),
    "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
    "MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD", "0.001",
    "DATASETS.TRAIN", "('epiphysis_r194_train_eval',)",
    "DATASETS.TEST", "('epiphysis_r194_train_eval',)",
    "INPUT.IMAGE_SIZE", "384",
    "INPUT.MIN_SCALE", "1.0",
    "INPUT.MAX_SCALE", "1.0",
    "INPUT.MASK_FORMAT", "bitmask",
    "TEST.DETECTIONS_PER_IMAGE", "100",
    "DATALOADER.NUM_WORKERS", "1",
    "OUTPUT_DIR", str(OUT / "maskdino_output"),
])
result = main(args)
summary = {
    "status": "eval_completed",
    "result": str(result),
    "constraints": {
        "eval_source": "R193 train subset, original TSRS_RSNA-Epiphysis/train first 24",
        "clean_test_v2_used": False,
        "new_or_reannotated_test_used": False,
        "success_claim_allowed": False
    }
}
(OUT / "r194_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

  cd "${MASKDINO_DIR}"
  "${PY}" "/home/shenzeyu/workspace/YOLO_SAM_generic_src/${OUT_DIR}/r194_eval_r193_on_train.py"
  cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

  "${PY}" scripts/diagnose_r192_maskdino_predictions.py \
    --coco-json "${R193_DIR}/coco/train.json" \
    --pred-json "${OUT_DIR}/maskdino_output/inference/coco_instances_results.json" \
    --out-json "${OUT_DIR}/r194_train_prediction_diagnostic.json"

  echo "[R194] done $(date -Is)"
} 2>&1 | tee "${LOG_DIR}/r194_maskdino_r193_train_diagnostic.log"
