#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

ENV_PREFIX="/home/shenzeyu/miniconda3/envs/aris-maskdino-r189-cu130"
PY="${ENV_PREFIX}/bin/python"
LOG_DIR="outputs/bridge_logs"
OUT_DIR="outputs/analysis/r190_maskdino_tiny_train_smoke"
MASKDINO_DIR="external_src/MaskDINO"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="/usr/local/cuda"
export PATH="${ENV_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/shenzeyu/workspace/YOLO_SAM_generic_src/${MASKDINO_DIR}:${PYTHONPATH:-}"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"

{
  echo "[R190] start $(date -Is)"
  echo "[R190] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

  "${PY}" -m pip install timm==1.0.22 shapely scipy
  "${PY}" scripts/prepare_r190_maskdino_coco_smoke.py \
    --raw-root data/raw \
    --dataset TSRS_RSNA-Epiphysis \
    --output-root "${OUT_DIR}/coco" \
    --train-count 12 \
    --val-count 4 \
    --max-size 384

  cat > "${OUT_DIR}/r190_register_and_train.py" <<'PY'
import json
import os
from pathlib import Path

from detectron2.data.datasets import register_coco_instances

ROOT = Path("/home/shenzeyu/workspace/YOLO_SAM_generic_src")
OUT = ROOT / "outputs/analysis/r190_maskdino_tiny_train_smoke"
COCO = OUT / "coco"
register_coco_instances("epiphysis_r190_train", {}, str(COCO / "train.json"), str(COCO / "train"))
register_coco_instances("epiphysis_r190_val", {}, str(COCO / "val.json"), str(COCO / "val"))

import sys
sys.path.insert(0, str(ROOT / "external_src/MaskDINO"))
from train_net import main
from detectron2.engine import default_argument_parser

args = default_argument_parser().parse_args([
    "--config-file", str(ROOT / "external_src/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"),
    "--num-gpus", "1",
    "MODEL.WEIGHTS", "",
    "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
    "MODEL.MaskDINO.NUM_OBJECT_QUERIES", "40",
    "MODEL.MaskDINO.DN_NUM", "10",
    "MODEL.MaskDINO.DEC_LAYERS", "2",
    "MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS", "1",
    "MODEL.MaskDINO.TRAIN_NUM_POINTS", "1024",
    "DATASETS.TRAIN", "('epiphysis_r190_train',)",
    "DATASETS.TEST", "('epiphysis_r190_val',)",
    "INPUT.IMAGE_SIZE", "384",
    "INPUT.MIN_SCALE", "1.0",
    "INPUT.MAX_SCALE", "1.0",
    "INPUT.MASK_FORMAT", "bitmask",
    "SOLVER.IMS_PER_BATCH", "1",
    "SOLVER.MAX_ITER", "12",
    "SOLVER.WARMUP_ITERS", "0",
    "SOLVER.BASE_LR", "0.0001",
    "SOLVER.STEPS", "()",
    "SOLVER.CHECKPOINT_PERIOD", "1000",
    "TEST.EVAL_PERIOD", "0",
    "DATALOADER.NUM_WORKERS", "0",
    "OUTPUT_DIR", str(OUT / "maskdino_output"),
])
args.eval_only = False
result = main(args)
summary = {
    "status": "train_smoke_completed",
    "result": str(result),
    "output_dir": str(OUT / "maskdino_output"),
}
(OUT / "r190_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

  cd "${MASKDINO_DIR}"
  "${PY}" "/home/shenzeyu/workspace/YOLO_SAM_generic_src/${OUT_DIR}/r190_register_and_train.py"
  cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

  echo "[R190] done $(date -Is)"
} 2>&1 | tee "${LOG_DIR}/r190_maskdino_tiny_train_smoke.log"
