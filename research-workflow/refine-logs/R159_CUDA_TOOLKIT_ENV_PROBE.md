# R159 CUDA Toolkit Environment Probe

## Purpose

R158 closed the current no-compile SegFormer branch. R159 checked whether the faithful external MaskDINO/Mask2Former branch can be reopened by creating or using an isolated CUDA-toolkit-capable environment.

This was a non-mutating probe. It did not install packages and did not launch training.

## Findings

Remote Python environment:

- Python path: `/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python`
- PyTorch: `2.5.1+cu118`
- CUDA runtime reported by PyTorch: `11.8`
- CUDA available: `true`
- GPU count: `2`

Compiler/tooling:

- `gcc`: available, version `13.3.0`
- `g++`: available, version `13.3.0`
- `nvcc`: not found
- `conda`: not found on shell `PATH`
- `mamba`: not found
- `micromamba`: not found

CUDA paths:

- `/usr/local/cuda`: exists
- `/usr/local/cuda-11.8`: absent
- `/usr/local/cuda-12*`: absent

Package availability:

- `pip index versions detectron2`: no matching distribution found
- `pip index versions fvcore`: available

## Decision

Faithful Detectron2/MaskDINO remains blocked/high-risk in the current remote shell environment.

Do not attempt Detectron2/MaskDINO installation into `yolo-sam-gpu`.

The blocker is stronger than just a missing Python module:

- no visible `nvcc`;
- no conda/mamba command on PATH for isolated CUDA-toolkit solve;
- no pip-visible Detectron2 wheel for the active Python/PyTorch stack;
- active environment uses PyTorch `2.5.1+cu118`, which is outside the common prebuilt Detectron2 wheel matrix.

## Next Valid Actions

1. User/admin action: provide an isolated environment with CUDA toolkit and `nvcc`, or expose a working conda/mamba command that can create one.
2. Continue only non-Detectron2/no-custom-CUDA architecture search, but require a strict overfit/smoke gate before clean-test-v2 diagnostics.
3. Return to data: obtain genuinely corrected train/val label PNGs, rebuild an isolated variant, and evaluate only on clean-test-v2/test.

## Evidence

- JSON probe: `outputs/analysis/r159_cuda_toolkit_env_probe.json`
- Log: `outputs/bridge_logs/r159_cuda_toolkit_env_probe.log`

