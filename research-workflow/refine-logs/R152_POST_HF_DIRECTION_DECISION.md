# R152 Post-HF Direction Decision

**Date**: 2026-07-02

**Target**: clean-test-v2 Dice `> 0.9317660066557425`

**Current valid best**: R110 clean-test-v2 Dice `0.9177231563529792`

## Decision

Close the current tiny-random HuggingFace Mask2Former route.

Do not run full HF Mask2Former union or instance training from the R144-R151 setup. The next branch should be a faithful external-implementation feasibility path, starting with R153 environment and install feasibility probing for official Detectron2-based Mask2Former / Mask DINO style code. If that probe shows dependency risk is too high, pivot to a different literature-backed architecture rather than tuning the current HF tiny configuration.

## Evidence

R144 proved only that the HF code path can consume the dataset. It did not prove model quality.

R145/R146 failed the first small-overfit gate:

- R145 loss dropped from `28.99` to `15.74`, but train/val/clean masks were empty.
- R146 bypassed class-score gating with mask-only readout, but masks were still empty.

R147/R148 diagnosed the failure:

- Foreground class scores were high (`~0.922`), but mask union probabilities collapsed to about `1e-5`.
- Thresholds `0.001` through `0.35` yielded Dice `0`.
- Ultra-low thresholds and GT-area top-k oracle still produced poor spatial ranking: train top-k oracle only `0.355161`.

R149/R149b showed the target path matters:

- Union target 1-image train Dice: `0.811855`.
- Instance target 1-image train Dice: `0.650195`.
- Instance target is not reliable enough for full training.

R150/R151 tested whether the union-source route is worth scaling:

- R150 1-image union overfit improved to Dice `0.891038`, Boundary IoU `0.215579`.
- R151 16-image small gate reached train Dice `0.857548`, but val Dice was only `0.710850`, Boundary IoU `0.079828`, and 4-image clean-test-v2 diagnostic Dice `0.639333`.

Therefore the HF tiny-random implementation is learnable but not competitive. It is far below the existing source-model band and nowhere near R110 or the target.

## Literature/Implementation Check

Mask2Former's official implementation is the `facebookresearch/Mask2Former` repository and is Detectron2-based. Its own getting-started material points users to Detectron2 setup.

Mask DINO's official implementation is `IDEA-Research/MaskDINO`; the repository states it is based on Detectron2 and its install path includes compiling the MSDeformAttn CUDA operator.

The server environment currently has:

- available: `torch`, `torchvision`, `timm`
- unavailable: `detectron2`, `fvcore`, `iopath`, `pycocotools`, `mmcv`, `mmengine`, `mmdet`, `monai`, `nnunetv2`

This means a faithful Mask2Former/Mask DINO branch is possible only as an isolated dependency effort. It should not mutate the existing `yolo-sam-gpu` environment.

## Options Considered

### Option A: Continue current HF tiny Mask2Former

Rejected. R151 already tested the strongest reasonable small gate from this setup and val Dice remained `0.710850`. More tuning of this random tiny config is unlikely to bridge the target gap.

### Option B: Faithful external Mask2Former / Mask DINO import

Preferred next branch. It aligns with the original R133/R143 literature motivation: real masked/deformable attention, high-resolution pixel embeddings, pretrained backbones, detection-style queries, and mature training code. The risk is dependency installation and dataset conversion cost.

### Option C: nnU-Net v2 / medical recipe

Lower priority now. R143 already tested a high-resolution medical recipe style source and ended at clean-test-v2 Dice `0.891375`, with high recall / low precision bridge failures. nnU-Net v2 remains a credible medical baseline, but it is less directly aligned with the instance-query failure mode and still requires data-format conversion.

### Option D: MedSAM / SAM-family return

Lower priority. Earlier SAM/MedSAM-style branches were high-recall complements or weak direct sources, and previous complement exploitation did not recover enough deployable signal.

## Next Gate

R153 should be a non-mutating faithful-external feasibility probe:

1. Record remote CUDA, PyTorch, Python, compiler, and GPU architecture state.
2. Check whether Detectron2 / Mask2Former / Mask DINO dependencies can be installed into an isolated target path or fresh env without touching `yolo-sam-gpu`.
3. Verify whether custom CUDA ops are likely buildable.
4. Produce a go/no-go artifact before any package installation or GPU training.

If R153 is feasible, R154 should create an isolated environment or dependency path, clone official code, convert 1-2 epiphysis examples to COCO/Detectron2-compatible format, and run a tiny train/infer smoke. Success evaluation must still use only `TSRS_RSNA-Epiphysis_clean_test_v2/test`, and only after train/val gates pass.

If R153 is infeasible, close the faithful Mask2Former/Mask DINO import path and pivot to another architecture family with a fresh smoke gate.
