# R187 External Environment Reprobe

R187 was executed on the remote server only.

## Purpose

Re-check whether a faithful external segmentation-model route is available after earlier MaskDINO/Mask2Former attempts were blocked by missing CUDA-toolkit / conda evidence.

## Server Evidence

- workspace: `/home/shenzeyu/workspace/YOLO_SAM_generic_src`
- current stable project Python: `/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python`
- current stable project torch: `2.5.1+cu118`
- visible GPUs: two RTX 4090 cards
- visible CUDA toolkit: `/usr/local/cuda/bin/nvcc`
- nvcc version: CUDA `13.0`
- conda: `/home/shenzeyu/miniconda3/bin/conda`
- `conda create --dry-run -n aris-maskdino-probe python=3.10` succeeded
- GitHub reachability passed for Mask2Former, MaskDINO, and Detectron2

## Decision

The faithful external-model route is reopened, but it must be isolated. Do not mutate the stable `yolo-sam-gpu` environment because its torch stack is `cu118` while the visible system CUDA toolkit is `13.0`.

Next action is R188: create a fresh isolated environment smoke for a faithful MaskDINO/Mask2Former/Detectron2-style route, then run only a tiny server-side smoke before any full training.

