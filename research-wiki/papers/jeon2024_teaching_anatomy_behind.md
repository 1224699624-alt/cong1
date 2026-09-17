---
type: paper
node_id: paper:jeon2024_teaching_anatomy_behind
title: "Teaching AI the Anatomy Behind the Scan: Addressing Anatomical Flaws in Medical Image Segmentation with Learnable Prior"
authors: ["Young Seok Jeon", "Hongfei Yang", "Huazhu Fu", "Mengling Feng"]
year: 2024
venue: "ICCV 2025"
external_ids:
  arxiv: "2403.18878"
  doi: null
  s2: null
tags: ["medical-segmentation", "anatomical-prior", "deformable-prior"]
added: 2026-07-27T04:42:28Z
---

# Teaching AI the Anatomy Behind the Scan: Addressing Anatomical Flaws in Medical Image Segmentation with Learnable Prior

## One-line thesis
Deforms a learnable canonical anatomical prior to patient-specific anatomy at global and local scales.

## Problem / Gap
_TODO._

## Method
_TODO._

## Key Results
_TODO._

## Assumptions
_TODO._

## Limitations / Failure Modes
_TODO._

## Reusable Ingredients
_TODO._

## Open Questions
_TODO._

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
_TODO._

## Abstract (original)

> Imposing key anatomical features, such as the number of organs, their shapes and relative positions, is crucial for building a robust multi-organ segmentation model. Current attempts to incorporate anatomical features include broadening the effective receptive field (ERF) size with data-intensive modules, or introducing anatomical constraints that scales poorly to multi-organ segmentation. We introduce a novel architecture called the Anatomy-Informed Cascaded Segmentation Network (AIC-Net). AIC-Net incorporates a learnable input termed "Anatomical Prior", which can be adapted to patient-specific anatomy using a differentiable spatial deformation. The deformed prior later guides decoder layers towards more anatomy-informed predictions. We repeat this process at a local patch level to enhance the representation of intricate objects, resulting in a cascaded network structure. AIC-Net is a general method that enhances any existing segmentation models to be more anatomy-aware. We have validated the performance of AIC-Net, with various backbones, on two multi-organ segmentation tasks: abdominal organs and vertebrae. For each respective task, our benchmarks demonstrate improved dice score and Hausdorff distance.

