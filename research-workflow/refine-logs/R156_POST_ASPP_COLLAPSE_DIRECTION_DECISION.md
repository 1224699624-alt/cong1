# R156 Post-ASPP Collapse Direction Decision

## Status

Target remains clean-test-v2 Dice `> 0.9317660066557425`.
Current valid best remains R110 clean-test-v2 Dice `0.9177231563529792`.

R155 tested a no-local-CUDA-compile ASPP/context decoder under the existing PyTorch/timm stack. It is not a viable quality branch in its current form:

- validation Dice `0.054529`
- validation Precision `0.028090`
- validation Recall `1.0`
- validation Specificity `0.0`
- validation Boundary IoU `0.0`
- clean-test-v2 four-image diagnostic Dice `0.038733`
- clean-test-v2 diagnostic Recall `1.0`
- clean-test-v2 diagnostic Specificity `0.0`

This is an all-foreground collapse, not a near-miss.

## Decision

Do not launch a full R155 run and do not run blind ASPP rate, threshold, image-size, or epoch sweeps.

The local lightweight architecture branch is now weakly exhausted:

- R143 high-resolution medical recipe stayed around clean-test-v2 Dice `0.891375`.
- R151 tiny HF Mask2Former union-source gate stayed far below source-model quality.
- R153 faithful Detectron2/MaskDINO path is blocked/high-risk without `nvcc`.
- R155 ASPP/context decoder collapsed to all foreground during the smoke gate.

## Next Valid Branches

1. **Faithful external architecture with proper environment**
   - Require an isolated CUDA-toolkit-capable environment with `nvcc`.
   - Then retry official Detectron2/Mask2Former or MaskDINO feasibility.
   - Do not mutate the existing `yolo-sam-gpu` environment.

2. **Different no-compile pretrained/literature route**
   - Must not be another plain U-Net/FPN/ASPP sweep.
   - Must pass a train-only overfit or tiny train/val smoke gate before any full run.
   - Success evaluation must remain only `TSRS_RSNA-Epiphysis_clean_test_v2/test`.

3. **Data/label correction with real corrected PNGs**
   - Only useful if new corrected train/val labels are supplied or produced.
   - Clean-test-v2 remains diagnostic/final evaluation only and cannot drive training decisions.

## Immediate Recommendation

Pause GPU launches and choose one of the two viable routes:

- create the isolated CUDA-toolkit environment needed for faithful MaskDINO/Mask2Former; or
- perform a new literature scan for a no-compile pretrained segmentation architecture with a materially different mechanism and a strict overfit gate.

