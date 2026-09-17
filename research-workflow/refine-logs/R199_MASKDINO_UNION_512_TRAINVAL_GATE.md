# R199 MaskDINO Union 512 Train/Val Gate

## Status

`RUNNING` on the remote server only.

## Purpose

R199 tests whether the promising R198 binary-union MaskDINO route benefits from higher-resolution input. It keeps the target as one foreground epiphysis union mask per image, but increases the COCO conversion and MaskDINO input size from `384` to `512`.

## Server-Only Constraint

- Training backend: `ssh shenzeyu@10.1.115.157`
- Remote workspace: `/home/shenzeyu/workspace/YOLO_SAM_generic_src`
- Environment: `/home/shenzeyu/miniconda3/envs/aris-maskdino-r189-cu130`
- GPU binding: `CUDA_VISIBLE_DEVICES=0`
- Screen: `r199_maskdino_union512_gpu0`
- Log: `outputs/bridge_logs/r199_maskdino_union_512_trainval_gate.log`

No local training was run.

## Data Policy

- Train: original `data/raw/TSRS_RSNA-Epiphysis/train`
- Val: original `data/raw/TSRS_RSNA-Epiphysis/val`
- Target transform: all positive instance ids become one binary foreground union mask.
- `clean-test-v2`: not used for training, tuning, or readout selection.
- New/reannotated test: not used.

## Launch

Launcher:

`outputs/bridge_logs/run_r199_maskdino_union_512_trainval_gate.sh`

Remote preflight showed both RTX 4090 GPUs idle. R199 was launched on GPU0 only. Initial monitoring confirmed active training with GPU0 memory use and Detectron2/MaskDINO iteration logs.

## Decision Gate

R199 is a train/val gate, not a success claim. After completion, compare val Dice, Boundary IoU, Boundary F1, separation-band false-positive rate, false-bridge flag, and component-count error against R198. Only if the val-selected readout is clearly stronger and stable should a later locked clean-test-v2 evaluation be considered.

## Monitoring

- [2026-07-04 02:00] Remote-only monitor: screen `r199_maskdino_union512_gpu0` is still active. Latest parsed metrics reached iter `2979/16000`, ETA about `87.7` minutes, total loss `16.7159`, learning rate `4e-05`. GPU0 uses about `2275 MiB`; GPU1 remains unused. `r199_summary.json`, prediction diagnostic JSON, val constraint JSON, COCO inference result JSON, and `model_final.pth` are not present yet. Log scan found no `Traceback`, `RuntimeError`, `CUDA out of memory`, or `Exception` markers.
- [2026-07-04 02:01] Remote-only monitor: screen `r199_maskdino_union512_gpu0` remains active. Latest parsed metrics reached iter `3199/16000`, ETA about `85.9` minutes, total loss `16.4128`, learning rate `4e-05`. GPU0 uses about `2275 MiB`; GPU1 remains unused. Final checkpoint and all R199 result JSON files are still absent, as expected before final train/val evaluation. Log scan found no error markers.
- [2026-07-04 02:02] Remote-only monitor: screen `r199_maskdino_union512_gpu0` remains active. Latest parsed metrics reached iter `3399/16000`, ETA about `84.6` minutes, total loss `13.0697`, learning rate `4e-05`. GPU0 uses about `2275 MiB`; GPU1 remains unused. Final checkpoint and all R199 result JSON files are still absent. Log scan found no `Traceback`, `RuntimeError`, `CUDA out of memory`, or `Exception` markers.
- [2026-07-04 02:04] Remote-only monitor: screen `r199_maskdino_union512_gpu0` remains active. Latest parsed metrics reached iter `3599/16000`, ETA about `83.3` minutes, total loss `14.2559`, learning rate `4e-05`. GPU0 uses about `2275 MiB`; GPU1 remains unused. Final checkpoint and all R199 result JSON files are still absent. Log scan found no error markers.
- [2026-07-04 02:05] Remote-only monitor: screen `r199_maskdino_union512_gpu0` remains active. Latest parsed metrics reached iter `3859/16000`, ETA about `81.3` minutes, total loss `13.8089`, learning rate `4e-05`. GPU0 uses about `2275 MiB`; GPU1 remains unused. R198 comparison files are visible on the remote server, while R199 final checkpoint and result JSON files are still absent. Log scan found no `Traceback`, `RuntimeError`, `CUDA out of memory`, or `Exception` markers.
- [2026-07-04 02:07] Remote-only monitor: screen `r199_maskdino_union512_gpu0` remains active. Latest parsed metrics reached iter `4119/16000`, ETA about `79.8` minutes, total loss `14.9241`, learning rate `4e-05`. GPU0 uses about `2275 MiB`; GPU1 remains unused. R199 final checkpoint and result JSON files are still absent. Log scan found no `Traceback`, `RuntimeError`, `CUDA out of memory`, or `Exception` markers.
- [2026-07-04 02:09] Remote-only monitor: screen `r199_maskdino_union512_gpu0` remains active. Latest parsed metrics reached iter `4359/16000`, ETA about `78.5` minutes, total loss `16.2856`, learning rate `4e-05`. GPU0 uses about `2275 MiB`; GPU1 remains unused. R199 final checkpoint and result JSON files are still absent. Log scan found no `Traceback`, `RuntimeError`, `CUDA out of memory`, or `Exception` markers.
- [2026-07-04 02:10] Remote-only monitor: screen `r199_maskdino_union512_gpu0` remains active. Latest parsed metrics reached iter `4559/16000`, ETA about `77.2` minutes, total loss `14.7515`, learning rate `4e-05`. GPU0 uses about `2275 MiB`; GPU1 remains unused. R199 final checkpoint and result JSON files are still absent. Log scan found no `Traceback`, `RuntimeError`, `CUDA out of memory`, or `Exception` markers.
- [2026-07-04 02:11] Remote-only monitor: screen `r199_maskdino_union512_gpu0` remains active. Latest parsed metrics reached iter `4739/16000`, ETA about `75.9` minutes, total loss `14.3556`, learning rate `4e-05`. GPU0 uses about `2275 MiB`; GPU1 remains unused. R199 final checkpoint and result JSON files are still absent. Log scan found no `Traceback`, `RuntimeError`, `CUDA out of memory`, or `Exception` markers.
