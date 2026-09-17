---
type: paper
node_id: paper:wang2025_layer_separation_joint_space
title: "Layer Separation: Towards Adjustable Joint Space Width Images Synthesis"
authors: ["Haolin Wang", "Yafei Ou", "Prasoon Ambalathankandy", "Gen Ota", "Pengyu Dai", "Masayuki Ikebe", "Kenji Suzuki", "Tamotsu Kamishima"]
year: 2025
venue: "ACM Multimedia 2025"
external_ids:
  arxiv: "2502.01972"
  doi: "10.1145/3746027.3755407"
  s2: null
tags: ["bone-overlap", "x-ray", "layer-separation", "joint-space", "ACM-MM-2025"]
added: 2026-08-03T12:00:00Z
---

# Layer Separation: Towards Adjustable Joint Space Width Images Synthesis

## One-line thesis
Separating upper-bone, lower-bone, and soft-tissue radiographic layers enables controllable joint-space synthesis and provides explicit supervision for projection overlap.

## Problem / Gap
Finger-joint radiographs contain superposed bone and soft-tissue signals, while RA datasets have limited and imbalanced joint-space-width variation.

## Method
The Layer Separation Networks decompose a conventional radiograph into upper-bone, lower-bone, and soft-tissue layers, then alter their relative positions and reconstruct radiographs with adjustable joint-space width.

## Key Results
The paper reports realistic synthesized radiographs and improvement in downstream tasks. It was published as a regular paper at ACM Multimedia 2025.

## Assumptions
The radiographic projection can be represented by anatomically meaningful constituent layers and recombined after controlled geometric changes.

## Limitations / Failure Modes
Its primary output is an image-layer decomposition and synthetic radiograph, not a complete per-bone multi-label mask. A realistic reconstructed X-ray does not by itself guarantee correct hidden instance boundaries.

## Reusable Ingredients
Bone/soft-tissue layer supervision, synthetic control of overlap severity, reconstruction consistency, and curriculum generation across joint-space widths.

## Open Questions
Whether separated layer features can supervise RAM-W600 amodal per-bone masks while retaining exact visible anatomy.

## Relevance to This Project
This is one of the closest peer-reviewed papers to RAM-W600 because it studies conventional RA finger-joint radiographs and explicitly manipulates overlap/joint space. It supports synthetic-overlap training and projection consistency, but requires an additional multi-label instance decoder for the project target.

## Local code audit and RAM test

The released generator accepts image plus bone masks and predicts three mask-supported layers. The current `random_movement` implementation has its affine transformation block commented out and only applies the context mask, so the repository cannot be assumed to reproduce the paper's random-shift curriculum without repair. R326 retained the transmission reconstruction but replaced GT inference masks with frozen nnU-Net probabilities and expanded two bone layers to fourteen. Its projection prior improved MSD slightly over a capacity-matched adapter, while Overlap DSC/IoU/NSD were slightly worse, indicating that equal optical-density pseudo layers are not yet sufficiently identifiable for overlap recovery.

## Primary Sources

- DOI: https://doi.org/10.1145/3746027.3755407
- Official ACM MM 2025 accepted-paper list: https://acmmm2025.org/accepted-regular-papers/
- Preprint: https://arxiv.org/abs/2502.01972
