#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

ENV_PREFIX="/home/shenzeyu/miniconda3/envs/aris-maskdino-r189-cu130"
PY="${ENV_PREFIX}/bin/python"
LOG_DIR="outputs/bridge_logs"
OUT_DIR="outputs/analysis/r191_maskdino_full_trainval_gate"
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
  echo "[R191] start $(date -Is)"
  echo "[R191] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

  "${PY}" scripts/prepare_r191_maskdino_full_coco.py \
    --raw-root data/raw \
    --dataset TSRS_RSNA-Epiphysis \
    --output-root "${OUT_DIR}/coco" \
    --train-count 0 \
    --val-count 0 \
    --max-size 384

  cat > "${OUT_DIR}/r191_register_and_train.py" <<'PY'
import json
from pathlib import Path

from detectron2.data.datasets import register_coco_instances

ROOT = Path("/home/shenzeyu/workspace/YOLO_SAM_generic_src")
OUT = ROOT / "outputs/analysis/r191_maskdino_full_trainval_gate"
COCO = OUT / "coco"
register_coco_instances("epiphysis_r191_train", {}, str(COCO / "train.json"), str(COCO / "train"))
register_coco_instances("epiphysis_r191_val", {}, str(COCO / "val.json"), str(COCO / "val"))

import sys
sys.path.insert(0, str(ROOT / "external_src/MaskDINO"))
from train_net import main
from detectron2.engine import default_argument_parser

args = default_argument_parser().parse_args([
    "--config-file", str(ROOT / "external_src/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"),
    "--num-gpus", "1",
    "MODEL.WEIGHTS", "",
    "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "1",
    "MODEL.MaskDINO.NUM_OBJECT_QUERIES", "80",
    "MODEL.MaskDINO.DN_NUM", "20",
    "MODEL.MaskDINO.DEC_LAYERS", "2",
    "MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS", "1",
    "MODEL.MaskDINO.TRAIN_NUM_POINTS", "2048",
    "MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD", "0.05",
    "DATASETS.TRAIN", "('epiphysis_r191_train',)",
    "DATASETS.TEST", "('epiphysis_r191_val',)",
    "INPUT.IMAGE_SIZE", "384",
    "INPUT.MIN_SCALE", "1.0",
    "INPUT.MAX_SCALE", "1.0",
    "INPUT.MASK_FORMAT", "bitmask",
    "SOLVER.IMS_PER_BATCH", "1",
    "SOLVER.MAX_ITER", "300",
    "SOLVER.WARMUP_ITERS", "0",
    "SOLVER.BASE_LR", "0.0001",
    "SOLVER.STEPS", "()",
    "SOLVER.CHECKPOINT_PERIOD", "300",
    "TEST.EVAL_PERIOD", "300",
    "TEST.DETECTIONS_PER_IMAGE", "80",
    "DATALOADER.NUM_WORKERS", "2",
    "OUTPUT_DIR", str(OUT / "maskdino_output"),
])
args.eval_only = False
try:
    result = main(args)
    status = "train_and_eval_completed"
    error = None
except Exception as exc:
    result = None
    status = "failed"
    error = repr(exc)
metrics_path = OUT / "maskdino_output" / "metrics.json"
metric_lines = []
if metrics_path.exists():
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                metric_lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
summary = {
    "status": status,
    "error": error,
    "result": str(result),
    "metrics_count": len(metric_lines),
    "last_metrics": metric_lines[-1] if metric_lines else None,
    "model_final_exists": (OUT / "maskdino_output" / "model_final.pth").exists(),
    "constraints": {
        "train_source": "data/raw/TSRS_RSNA-Epiphysis/train",
        "val_source": "data/raw/TSRS_RSNA-Epiphysis/val",
        "clean_test_v2_used": False,
        "new_or_reannotated_test_used": False,
        "success_claim_allowed": False
    }
}
(OUT / "r191_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if status == "failed":
    raise SystemExit(2)
PY

  cd "${MASKDINO_DIR}"
  "${PY}" "/home/shenzeyu/workspace/YOLO_SAM_generic_src/${OUT_DIR}/r191_register_and_train.py"
  cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

  echo "[R191] done $(date -Is)"
} 2>&1 | tee "${LOG_DIR}/r191_maskdino_full_trainval_gate.log"

