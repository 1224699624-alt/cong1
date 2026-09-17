# R214 Spatial-Aware Topology GitHub Code Audit

Date: 2026-07-07

## Source

Repository inspected in an isolated local directory:

- `.tmp/Topology-Preserving-Image-Segmentation-with-Spatial-Aware-Persistent-Feature-Matching`
- Upstream: `https://github.com/josiahmutie-pixel/Topology-Preserving-Image-Segmentation-with-Spatial-Aware-Persistent-Feature-Matching.git`

## What The Repository Actually Contains

The repository is notebook-based, not a reusable Python package. There is no
real `src/` implementation despite the README tree. The core code is in:

- `01_road_segmentation_satloss.ipynb`
- `02_crack_segmentation_cldice_satloss.ipynb`
- `03_celegans_transfer_learning.ipynb`
- `04_drive_retinal_vessel_transfer_learning.ipynb`

The method is built for thin connected foreground tasks:

- roads;
- cracks;
- retinal vessels;
- C. elegans foreground transfer.

This task bias matters because epiphysis segmentation often needs separated
neighboring bones, not stronger foreground connectivity.

## Core Mechanism

The useful idea is not the U-Net backbone. The useful idea is the SATLoss
candidate mechanism:

1. Use `gudhi.CubicalComplex(top_dimensional_cells=-prob_np)` on a detached
   probability map.
2. Extract persistence pairs from dimensions 0 and 1.
3. Convert creator/destroyer flat indices to image coordinates.
4. Match predicted and GT persistence features greedily with a spatial-aware
   cost:
   - diagram/value difference;
   - creator/destroyer coordinate distance.
5. Backpropagate only through the selected prediction birth/death pixels.

Important implementation detail: persistence extraction is not differentiable;
gradients only flow through the chosen pixels after feature extraction.

## Directly Useful Ideas For YOLO+SAM Epiphysis

### 1. Spatially aware topology diagnostics

The repository's creator/destroyer coordinates are useful as diagnostics. For
our task, compare topology events inside bone-gap / adjacent-bone regions
rather than over the whole foreground.

Candidate use:

- locate high-persistence false bridges;
- rank bridge-risk pixels by birth/death coordinates;
- add these coordinates to hard-case visualization;
- add PH-derived features to R214/R215 candidate rows.

### 2. Candidate-level image/context features

The matching idea supports the current R214 direction: candidate gates need
spatial and local image context, not only shape/contact features. Useful
features include:

- cut birth/death coordinate normalized by image size;
- distance from cut to local component boundary;
- local image gradient/intensity at cut, ring, and bbox;
- persistence-like score of a neck/bridge candidate.

### 3. Creator-destroyer logic as a bridge suppressor

The repo's destroyer term pushes unmatched predicted topological features
toward the diagonal. In our task this should be adapted as:

- suppress unmatched predicted bridge components inside GT-free gap-risk
  bands;
- avoid creator-only losses that encourage missing connections;
- keep component-preserving guards so the model does not over-erode.

## What Should Not Be Copied Directly

Do not directly use the repo's clDice/SATLoss as a training objective for the
current mainline.

Reasons:

- clDice is connectivity-preserving; epiphysis needs separation-preserving
  behavior.
- Global beta0/beta1 matching can reward the wrong behavior if adjacent bones
  should stay disconnected.
- The notebook results are mixed: SATLoss does not consistently improve Dice
  or topology versus hybrid baselines.
- The implementation depends on `gudhi` and CPU-side detached persistence
  extraction, which is slow and awkward for full-resolution training.
- It is notebook-quality code, not a stable library.

## Recommended Project Integration

Use it as inspiration for a guarded validation-only experiment, not as a full
training replacement.

### R214

Continue image-context candidate features in
`scripts/run_r211_f1_learned_acceptance_gate.py`.

Purpose:

- improve useful/risk separation for R110-anchored local edits;
- keep all training/selection on original train/val;
- no clean-test-v2 tuning.

### R215

Add a PH-inspired candidate diagnostic script:

- input: R110 anchor mask, candidate cut, image, optional original train/val GT;
- output: candidate CSV features only;
- dependency gate: check if `gudhi` is available on the remote env before using
  it in any training loop.

Potential features:

- candidate persistence rank within local bbox;
- birth/death coordinates normalized to local bbox and image;
- matched/unmatched persistence count before/after cut;
- local beta0 change inside gap-risk band;
- local hole count change near candidate.

Promotion rule:

- only use PH features if grouped image validation improves useful-recall at a
  fixed hard-risk budget;
- otherwise keep PH as visualization/audit only.

## Bottom Line

This GitHub repo is valuable as a conceptual source for spatially localized
topology events, but it should not be transplanted as-is. For our objective,
the right adaptation is separation-aware, local, candidate-level topology
features and diagnostics, with R201 metrics and hard-case visual checks.
