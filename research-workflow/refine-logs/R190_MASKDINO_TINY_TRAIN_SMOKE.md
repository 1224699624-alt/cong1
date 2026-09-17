# R190 MaskDINO Tiny Train Smoke

R190 was executed on the remote server only.

## Purpose

Verify that the faithful external architecture route can actually train on this project after R189 opened a CUDA-13.0-compatible Detectron2/MaskDINO environment.

## Setup

- environment: `aris-maskdino-r189-cu130`
- GPU binding: one RTX 4090 via `CUDA_VISIBLE_DEVICES=0`
- data: original `TSRS_RSNA-Epiphysis` train/val only
- smoke dataset: 12 train images / 4 val images converted to COCO instance format
- image size: 384
- classes: 1 (`epiphysis`)
- clean-test-v2: not used
- new/reannotated test: not used

Server-side compatibility fixes required:

- used torch `2.11.0+cu130` with CUDA `13.0`
- compiled Detectron2 successfully
- patched MaskDINO `MultiScaleDeformableAttention` CUDA op for modern PyTorch dispatch (`value.scalar_type()` / `value.is_cuda()`)
- installed and verified `cv2`
- patched MaskDINO COCO mapper to accept RLE bitmask annotations and convert `BitMasks` to tensor masks

## Result

The tiny MaskDINO training smoke reached forward/backward training and saved a checkpoint:

- status: `train_smoke_completed_eval_topk_failed`
- final logged iteration: `11`
- total loss: `56.385128289461136`
- model checkpoint: `outputs/analysis/r190_maskdino_tiny_train_smoke/maskdino_output/model_final.pth`
- checkpoint size: `354323613` bytes
- metrics: `outputs/analysis/r190_maskdino_tiny_train_smoke/maskdino_output/metrics.json`

The automatic post-training evaluator failed after training because the smoke used only 40 queries while the default inference top-k expected 100 detections. This is an evaluation configuration issue, not a training failure.

## Decision

The faithful MaskDINO route is now technically trainable on the server. R191 should convert the full original train/val set to COCO instance format and run a guarded small-scale MaskDINO overfit/validation gate with `TEST.DETECTIONS_PER_IMAGE <= NUM_OBJECT_QUERIES`, still without using clean-test-v2 for tuning.

