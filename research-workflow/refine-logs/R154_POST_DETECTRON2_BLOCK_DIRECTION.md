# R154 Post-Detectron2-Block Direction

**Date**: 2026-07-02

**Target**: clean-test-v2 Dice `> 0.9317660066557425`

## Decision

Do not attempt official Detectron2-based Mask2Former / Mask DINO installation in the current remote environment.

The next GPU branch should be a no-local-CUDA-compile architecture implemented under the existing PyTorch/timm stack, but it must not be another plain U-Net/medical recipe sweep. The selected next branch is R155: a SegFormer/DeepLabv3+-style ASPP/context decoder with boundary/distance supervision and conservative validation gate.

## Why Detectron2/MaskDINO Is Blocked

R153 was non-mutating and found:

- PyTorch `2.5.1+cu118`, CUDA runtime `11.8`, two RTX 4090 GPUs.
- GitHub reachable.
- gcc/g++ available.
- `nvcc` unavailable.
- `detectron2`, `fvcore`, `iopath`, `pycocotools`, and `ninja` unavailable.
- Disk free about `167G`, but root filesystem already `91%` used.

Official Detectron2-based Mask2Former/MaskDINO paths usually require compiled dependencies and/or custom CUDA ops. Without `nvcc`, this path is high-risk and should not be attempted inside `yolo-sam-gpu`.

## Why Not Continue Existing Routes

- Current HF tiny random Mask2Former: closed by R151. It learned a weak union source but val Dice was only `0.710850`.
- High-resolution U-Net medical recipe: R143 already reached clean-test-v2 Dice only `0.891375` with bridge/overmask failures.
- DINOv3 instance-separation sweeps: R130/R140 showed structure improvements but clean-test-v2 stayed around `0.885-0.891`.
- Lightweight set/query models: R129/R132 overmasked and stayed far below R110.
- Same-family patch/readout/fusion: exhausted around R110/R131 with non-leaking oracle near `0.919`, far below target.

## Selected Next Branch: R155 ASPP/Context Decoder

R155 should test a different pure-PyTorch decoder class before giving up on local architectures:

- timm pretrained encoder, preferably ConvNeXt-tiny or DINOv3-backed if available.
- ASPP / multi-dilation context head similar in spirit to DeepLabv3+.
- Shallow high-resolution refinement skip.
- Boundary and signed-distance supervision reused from existing scripts.
- Conservative validation selection over threshold and min-component cleanup.
- Smoke first, then full only if smoke passes.

This branch is still a long shot, but it is cheap, does not require custom CUDA compilation, and differs from the existing FPN/U-Net decoder enough to be a valid mechanism test.

## R155 Gate

R155 should start with a small smoke:

- `limit-train <= 16`, `limit-val <= 8`, `limit-eval <= 4`
- fixed evaluation protocol
- clean-test-v2 only after the smoke train/val run completes
- no success claim from smoke clean-test-v2

Only if smoke shows reasonable train/val learning should a full R155 run be launched. If the full R155 clean-test-v2 result remains in the broad `0.88-0.90` source-model band, close local pure-PyTorch source-model architecture search and return to data/label strategy or external environment provisioning.

## Requirements Before Any Future Faithful MaskDINO Retry

To reopen official MaskDINO/Mask2Former:

- isolated environment, not `yolo-sam-gpu`
- CUDA toolkit / `nvcc` available
- `ninja`, `pycocotools`, `fvcore`, `iopath`, `detectron2` installed in isolation
- successful import and custom-op smoke
- 1-2 image train/infer smoke before any full run
