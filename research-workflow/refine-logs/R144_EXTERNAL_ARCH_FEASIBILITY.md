# R144 External Architecture Feasibility Plan

**Date**: 2026-07-01

**Reason**: R143 high-resolution medical-recipe scaling failed with clean-test-v2 Dice `0.891375`, far below R110 `0.917723` and target `0.931766`.

## Decision

R144 should test feasibility of a real external instance-segmentation architecture rather than another local lightweight approximation.

Preferred family:

- Mask2Former / Mask DINO style masked or deformable attention with query masks, no-object calibration, high-resolution pixel embeddings, and instance-level supervision.

Not acceptable as R144:

- another `timm` U-Net resolution sweep,
- another R132-style lightweight query head,
- another R140 reviewed-variant retrain,
- clean-test-v2-tuned postprocessing.

## Current Remote Dependency State

Present:

- `torch`
- `torchvision`
- `timm`
- `segment_anything`

Missing:

- `detectron2`
- `mmdet`
- `mmcv`
- `mmengine`
- `transformers`
- `monai`
- `nnunetv2`

## R144 Minimum Smoke

Before any full GPU run:

1. Create an isolated dependency path or conda clone. Do not break `yolo-sam-gpu`.
2. Convert 2 train images and their instance-valued labels into the selected framework format.
3. Run a 2-image overfit/smoke for at least one optimizer step.
4. Run inference on 2 clean-test-v2 images.
5. Export binary union masks and metrics JSON under `outputs/analysis/`.
6. Confirm clean-test-v2 is only used for smoke/final eval, not training/tuning.

## Full-Run Gate

Only launch full R144 if smoke passes and the framework can:

- consume variable instance counts,
- preserve instance separation in training,
- export binary union masks compatible with `evaluate_masks.py`,
- run on one free GPU without occupying both cards.

## Stop Rule

If dependency installation becomes the dominant task or requires risky changes to the working conda env, stop and document the blocker. Do not half-install packages into the active experiment environment.

## Smoke Result

Status: `DONE_SMOKE_POSITIVE`

Artifacts:

- Script: `scripts/r144_mask2former_hf_smoke.py`
- Launcher: `outputs/bridge_logs/run_r144_mask2former_hf_smoke.sh`
- Metrics: `outputs/analysis/r144_mask2former_hf_smoke_clean_test_v2_metrics.json`
- Manifest: `outputs/analysis/r144_mask2former_hf_smoke_manifest.json`
- Log: `outputs/bridge_logs/r144_mask2former_hf_smoke.log`

What passed:

- Installed `transformers==4.53.0` into isolated project path `.tmp/r144_hf_deps`.
- Did not mutate the active `yolo-sam-gpu` conda environment.
- Constructed a real `Mask2FormerForUniversalSegmentation` model.
- Converted instance-valued PNG labels into variable-instance `mask_labels` tensors.
- Ran 2 optimizer steps on 2 original train images.
- Ran inference on 2 clean-test-v2 images.
- Exported binary union masks and metrics JSON.
- Preserved clean-test-v2 as smoke/final inference only, not training or tuning.

Observed smoke losses:

- Step 1: `19.300716400146484`
- Step 2: `19.105274200439453`

Smoke metrics are intentionally not meaningful as quality evidence because the model is tiny/random and trained for only two steps. The mean clean-test-v2 Dice over the 2 smoke images was `0.000047059561718615196`.

## Next Gate

Proceed to R145, not a full run yet.

R145 should test whether this external Mask2Former path can overfit a small epiphysis subset and produce non-empty useful masks:

1. Train on 8-16 original train images for enough steps to show loss decrease and train-set Dice recovery.
2. Evaluate on a tiny original val subset and 2-8 clean-test-v2 images for diagnostics only.
3. Keep dependencies isolated under `.tmp/r144_hf_deps` unless a separate environment is created.
4. Launch a full-dataset run only if small-overfit behavior is real.

## R145 / R146 Follow-Up

R145 ran the small-overfit gate with 16 train images, 4 val images, and 4 clean-test-v2 diagnostic images. Loss decreased from `28.99` to `15.74` over 120 steps, so optimization is active. However, exported masks were empty at threshold `0.35`: train Dice `0`, val Dice `0`, and clean-test-v2 Dice `0`.

R146 bypassed the class-score foreground gate with `--readout-mode mask_only`. It still produced empty train/val/clean masks. Therefore the immediate blocker is not only no-object/class gating; the mask logits or probability scale must be inspected before any full external-architecture run.

Next required diagnostic: R147 should load the R146 checkpoint and sweep mask-union thresholds, recording max/quantile probabilities and Dice by threshold on the same tiny train/val/clean subsets.
