# R143 Direction Decision After R140/R141

**Date**: 2026-07-01

**Target**: clean-test-v2 Dice `> 0.9317660066557425`

**Current best valid result**: R110 Dice `0.9177231563529792`

## Evidence State

R140 tested the reviewed-data branch on the isolated reviewed variant `TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1`. It is a clear negative:

- Best validation Dice: `0.8763645049058203`
- clean-test-v2 Dice: `0.8855000712919942`
- Delta vs R110: `-0.032223085060985035`
- Gap to target: `0.046265935363748345`

R141 then checked whether R140 is at least complementary to R110:

- R140 is better on only `2/81` clean-test-v2 cases.
- Per-image oracle over R110/R140: `0.9177697105754016`
- Gain vs R110: `0.00004655422242239027`

Decision: close the current reviewed-variant source-model branch. Do not launch R140 fusion, R140 threshold sweeps, or another DINOv3 instance-separation run on the same reviewed variant.

## Literature-Grounded Options

### Option A: Full Mask2Former / Mask DINO Integration

Mask2Former is relevant because its main mechanism is masked attention: cross-attention is constrained within predicted mask regions, which is different from the lightweight query-head approximations already tested in R129/R132. Mask DINO is relevant because it extends detection-style DINO queries with mask prediction over high-resolution pixel embeddings and explicit detection/segmentation losses.

This branch is scientifically clean but engineering-heavy. The remote environment currently has:

- Present: `torch`, `torchvision`, `timm`, `segment_anything`
- Missing: `detectron2`, `mmdet`, `mmcv`, `mmengine`, `transformers`, `monai`, `nnunetv2`

Implication: do not pretend R132 was a full Mask2Former/Mask DINO test, but also do not start a long GPU run before dependency installation and a tiny data-format smoke pass.

### Option B: Medical Recipe Pivot

nnU-Net remains relevant because it argues for strong, rigorously configured medical segmentation baselines before claiming exotic architecture gains. For this 2D X-ray task, the practical translation is a high-resolution 2D U-Net-style baseline with robust augmentation, deep supervision or multiscale outputs, careful foreground sampling, validation-selected thresholding, and optional connected-component cleanup.

This branch is less novel but cheaper and compatible with the current PyTorch/timm environment. It tests whether the gap is from training recipe understrength rather than missing transformer machinery.

### Option C: Anatomical Graph/Layout Constraint

This is conceptually aligned with the failure mode, but previous layout-only attempts R074/R075 lost recall and R128 showed instance ID/order is too noisy for naive fixed slots. Do not launch this as R143 unless a new non-leaking train/val diagnostic shows a concrete layout rule with positive validation Dice.

## Decision

Choose Option B as the immediate R143 implementation branch, and keep Option A as the heavier follow-up if R143 fails.

Reason:

1. R140/R141 closed the reviewed-data shortcut.
2. R132 closed only a lightweight query approximation, not full Mask2Former/Mask DINO.
3. The server is not ready for detectron2/MMDetection-style integration today.
4. A strong 2D medical segmentation recipe is the cheapest remaining way to test whether the current base predictors are undertrained relative to the dataset.

## R143 Proposed Run

Run ID: `R143`

Name: `r143_highres_medical_recipe_unet`

Dataset:

- Train/val: original `TSRS_RSNA-Epiphysis` train/val, not reannotated test.
- Final success evaluation: `TSRS_RSNA-Epiphysis_clean_test_v2/test` only.
- Original test may be a control metric only.

Mechanism:

- PyTorch high-resolution 2D segmentation model using the existing project dataset loaders where possible.
- Prefer a robust encoder/decoder already supported by the environment (`timm` encoder or plain U-Net if simpler).
- Add medical-recipe changes rather than another candidate fusion head:
  - high-resolution crop or resize policy,
  - strong but X-ray-safe intensity augmentation,
  - foreground-aware sampling,
  - BCE/Dice plus boundary or distance auxiliary loss,
  - validation-selected threshold,
  - optional small-component cleanup selected on validation only.

Smoke gate:

- 4 train images, 2 val images, 2 clean-test-v2 images.
- Must complete train -> val -> clean-test-v2 -> original-test control without crash.
- Must write metrics JSONs under `outputs/analysis/`.

Full-run gate:

- Launch only after smoke passes.
- Use one free GPU only.
- Stop early if validation Dice cannot reach the R110/R130 validation band after minimum epochs.

Success:

- New best if clean-test-v2 Dice `> 0.9177231563529792`.
- Target met only if clean-test-v2 Dice `> 0.9317660066557425`.

Failure interpretation:

- If R143 is far below R110, recipe strength is not the missing ingredient and the next branch should be full external Mask2Former/Mask DINO integration, starting with dependency installation and COCO/instance-format conversion smoke.

## Sources

- Mask2Former: https://arxiv.org/abs/2112.01527
- Mask DINO: https://arxiv.org/abs/2206.02777
- Mask DINO official implementation: https://github.com/IDEA-Research/MaskDINO
- nnU-Net: https://pubmed.ncbi.nlm.nih.gov/33288961/
- nnU-Net Revisited: https://arxiv.org/html/2404.09556v2
