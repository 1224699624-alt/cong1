# R188 External Model Environment Smoke

R188 was executed on the remote server only.

## What Passed

An isolated conda environment `aris-maskdino-r188` was created on the server. PyTorch CUDA smoke passed on GPU0:

- torch: `2.5.1+cu124`
- torch CUDA: `12.4`
- CUDA available: `true`
- visible device count under binding: `1`
- GPU tensor forward/backward smoke: `ok`
- device: `NVIDIA GeForce RTX 4090`

MaskDINO and Detectron2 sources were cloned under `external_src/`.

## What Failed

Detectron2 failed to build because the server compiler toolkit and PyTorch CUDA runtime did not match:

- detected CUDA toolkit / nvcc: `13.0`
- PyTorch build CUDA: `12.4`

The failure is therefore an environment-stack mismatch, not a dataset or training failure.

## Decision

Do not train with R188. Proceed to R189 with an isolated CUDA-13.0-matched PyTorch stack, using the official `cu130` wheels, then retry the Detectron2 import/operator smoke.

