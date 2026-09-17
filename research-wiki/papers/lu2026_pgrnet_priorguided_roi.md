---
type: paper
node_id: paper:lu2026_pgrnet_priorguided_roi
title: "PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation"
authors: ["Jiacheng Lu", "Hui Ding", "Shiyu Zhang", "Guoping Huo"]
year: 2026
venue: "CVPR 2026"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["medical-segmentation", "spatial-prior", "roi", "top-k", "brain-tumor", "cvpr-2026"]
added: 2026-07-15T10:40:49Z
---

# PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation

## One-line thesis
Training-mask-derived spatial and scale ROI templates are dynamically selected across feature layers and converted into soft Gaussian-decay guidance maps for efficient lesion-focused segmentation.

## Problem / Gap

Brain-tumor pixels occupy only about 10.7% of a standardized 160x160 BraTS slice, while conventional segmenters spend most feature capacity on background and ignore the non-uniform spatial distribution of lesions.

## Method

1. Extract connected components from training masks and summarize their normalized bounding-box centers and scale modes as a finite ROI template set.
2. Score the candidate templates on encoder features and progressively retain high-confidence candidates with a Hierarchical Top-K decision mechanism.
3. Convert selected candidates into WinGS-ROI maps: confidence-weighted circular Gaussian maps with an additional decay outside each ROI boundary.
4. Use the maps to modulate encoder features, ROI-local RetNet computation, skip connections, and decoder upsampling; fall back to full-image processing when ROI confidence is unreliable.
5. Train the ROI scorers indirectly and end-to-end using only the segmentation loss (paper: Dice:BCE = 2:8).

## Key Results

- BraTS-2019 / BraTS-2023 / MSD Task01 Whole-Tumor Dice: 89.02% / 91.82% / 89.67%.
- Full-model ablation on BraTS-2023 reports WT/TC/ET Dice of 91.82/94.07/93.88 and HD95 of 1.1334/0.6647/0.6011.
- Reported complexity: 8.64M parameters and 39.05G FLOPs.
- Paper reports full-image fallback in 6.97%, 3.52%, and 5.33% of cases on the three datasets.

## Assumptions

- Images share a meaningful canonical coordinate system after cropping/resizing to 160x160.
- Training-mask component centers and scales approximate the test distribution.
- A small finite set of roughly circular ROIs can cover the target lesion distribution.
- Dataset-level training labels are reliable enough to construct candidate templates.

## Limitations / Failure Modes

- Direct spatial coordinates do not transfer to unregistered pediatric hand radiographs with variable pose, field of view, scale, laterality, age, and sex.
- A single tumor-like ROI decision is not an instance model for multiple adjacent bones or multiple required seams.
- The released code materially differs from the paper: it does not implement the stated recursive cross-layer candidate filtering, confidence matrix, entropy/gap fallback, or final single-ROI lock.
- `ROIScorer.forward` averages scores across the batch, so ROI decisions are batch-level rather than per-image.
- The provided training configuration uses `retention_at='bottleneck'`, which bypasses the e3 HTK/WinGS/ROI-RetNet branch guarded by `self.ret_enc3 is not None`.
- The released prior-generation script clusters centers and uses maximum cluster size, retains tiny 1-2 pixel candidates, and does not reproduce the paper's scale-peak construction exactly.
- The code compares an uncalibrated raw mean scorer logit against a 0.5 threshold and uses this as its fallback gate.

## Reusable Ingredients

- Learn a dataset-level prior from training masks only; no test-image GT is required at inference.
- Separate broad candidate coverage from image-conditioned candidate confidence.
- Use confidence-gated soft modulation plus a full-image fallback instead of hard cropping.
- Preserve the base segmenter's full-image path and add ROI guidance as a residual correction.
- For this project, replace circular tumor ROIs with canonical, anisotropic seam-strip templates and make selection per-image and per anatomical seam slot.

## Open Questions

- Can hand canonicalization be accurate enough to support normalized seam coordinates without leaking segmentation GT at inference?
- Should age and sex condition a continuous mixture of priors rather than discrete bins, given age imbalance?
- Can the ROI confidence be calibrated against seam visibility/contrast and used to suppress the new loss when the prior is uncertain?

## Claims

The paper supports the general claim that a training-derived spatial prior, selected by image features and applied softly across network stages, can improve overlap and boundary metrics. It does not establish that the released code exactly reproduces every paper mechanism, nor that the method directly solves multi-bone instance separation.

## Connections

Supports [[heterochrony_aware_prediction_derived_separation_prior]] as evidence for candidate-prior plus image-conditioned confidence and fallback, but motivates a canonical anatomical coordinate system and seam-specific templates rather than raw global ROIs.

## Relevance to This Project

High conceptual relevance, low direct-code portability. The most useful transfer is not the RetNet backbone but the three-part interface: conditional candidate prior -> per-image confidence selector -> soft residual guidance/fallback. Pediatric hands require age/sex-conditioned canonical seam priors, multiple named seam slots, anisotropic maps, and confidence-weighted separation/boundary losses.

