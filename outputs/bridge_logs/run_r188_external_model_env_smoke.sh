#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src

ENV_NAME="aris-maskdino-r188"
CONDA="/home/shenzeyu/miniconda3/bin/conda"
ENV_PREFIX="/home/shenzeyu/miniconda3/envs/${ENV_NAME}"
LOG_DIR="outputs/bridge_logs"
OUT_DIR="outputs/analysis/r188_external_model_env_smoke"
SRC_DIR="external_src"
mkdir -p "${LOG_DIR}" "${OUT_DIR}" "${SRC_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

{
  echo "[R188] start $(date -Is)"
  echo "[R188] gpu binding CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  df -h . /home/shenzeyu || true

  if [ ! -x "${ENV_PREFIX}/bin/python" ]; then
    echo "[R188] creating isolated conda env ${ENV_NAME}"
    "${CONDA}" create -y -n "${ENV_NAME}" python=3.10 pip
  else
    echo "[R188] reusing existing isolated env ${ENV_NAME}"
  fi

  PY="${ENV_PREFIX}/bin/python"
  "${PY}" -m pip install --upgrade pip setuptools wheel packaging ninja cython pyyaml tqdm pycocotools

  if ! "${PY}" - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
  then
    echo "[R188] installing isolated PyTorch CUDA stack"
    "${PY}" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
  fi

  "${PY}" - <<'PY'
import json, torch
payload = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    x = torch.randn(8, 3, 64, 64, device="cuda")
    conv = torch.nn.Conv2d(3, 4, 3, padding=1).cuda()
    y = conv(x).mean()
    y.backward()
    payload["gpu_tensor_smoke"] = "ok"
    payload["device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(payload, ensure_ascii=False, indent=2))
open("outputs/analysis/r188_external_model_env_smoke/torch_gpu_smoke.json", "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2))
PY

  if [ ! -d "${SRC_DIR}/detectron2/.git" ]; then
    git clone --depth 1 https://github.com/facebookresearch/detectron2.git "${SRC_DIR}/detectron2"
  else
    git -C "${SRC_DIR}/detectron2" rev-parse --short HEAD
  fi
  if [ ! -d "${SRC_DIR}/MaskDINO/.git" ]; then
    git clone --depth 1 https://github.com/IDEA-Research/MaskDINO.git "${SRC_DIR}/MaskDINO"
  else
    git -C "${SRC_DIR}/MaskDINO" rev-parse --short HEAD
  fi

  "${PY}" -m pip install -e "${SRC_DIR}/detectron2" --no-build-isolation

  "${PY}" - <<'PY'
import json, os, subprocess, sys
import torch
import detectron2
from detectron2.layers import nms
boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]], device="cuda" if torch.cuda.is_available() else "cpu")
scores = torch.tensor([0.9,0.8], device=boxes.device)
keep = nms(boxes, scores, 0.5).detach().cpu().tolist()
payload = {
    "status": "detectron2_import_ok",
    "detectron2_version": getattr(detectron2, "__version__", "unknown"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "nms_keep": keep,
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
open("outputs/analysis/r188_external_model_env_smoke/detectron2_smoke.json", "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2))
PY

  "${PY}" - <<'PY'
import json, pathlib
root = pathlib.Path("external_src/MaskDINO")
expected = [
    root / "maskdino",
    root / "train_net.py",
    root / "configs",
]
payload = {
    "status": "maskdino_source_ready" if all(p.exists() for p in expected) else "maskdino_source_incomplete",
    "existing": {str(p): p.exists() for p in expected},
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
open("outputs/analysis/r188_external_model_env_smoke/maskdino_source_smoke.json", "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2))
PY

  "${PY}" - <<'PY'
import json, pathlib
out = pathlib.Path("outputs/analysis/r188_external_model_env_smoke/r188_summary.json")
parts = {}
for name in ["torch_gpu_smoke", "detectron2_smoke", "maskdino_source_smoke"]:
    path = out.parent / f"{name}.json"
    if path.exists():
        parts[name] = json.loads(path.read_text(encoding="utf-8"))
status = "ready_for_r189_tiny_train_smoke"
if parts.get("detectron2_smoke", {}).get("status") != "detectron2_import_ok":
    status = "blocked_detectron2"
if parts.get("maskdino_source_smoke", {}).get("status") != "maskdino_source_ready":
    status = "blocked_maskdino_source"
summary = {"status": status, "parts": parts}
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

  echo "[R188] done $(date -Is)"
} 2>&1 | tee "${LOG_DIR}/r188_external_model_env_smoke.log"
