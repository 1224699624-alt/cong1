# R189 Detectron2 CUDA-13 Environment Smoke

R189 was executed on the remote server only.

## Purpose

Fix the R188 CUDA mismatch by using an isolated PyTorch stack compiled for CUDA `13.0`, matching the server's `/usr/local/cuda-13.0` toolkit.

## Result

R189 passed.

- isolated env: `aris-maskdino-r189-cu130`
- torch: `2.11.0+cu130`
- torch CUDA: `13.0`
- GPU tensor/backward smoke: passed on one RTX 4090
- Detectron2 build: passed
- Detectron2 import/NMS smoke: passed
- MaskDINO source: ready under `external_src/MaskDINO`

Summary artifact:

- `outputs/analysis/r189_detectron2_cu130_env_smoke/r189_summary.json`

## Decision

The faithful external architecture route is now technically open. Proceed to R190: a tiny MaskDINO/Detectron2 train smoke on original train/val only, still evaluating no success claim on clean-test-v2 until a locked configuration exists.

