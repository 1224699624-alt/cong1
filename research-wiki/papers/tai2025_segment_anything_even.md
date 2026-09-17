---
type: paper
node_id: paper:tai2025_segment_anything_even
title: "Segment Anything, Even Occluded"
authors: ["Wei-En Tai", "Yu-Lin Shih", "Cheng Sun", "Yu-Chiang Frank Wang", "Hwann-Tzong Chen"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2503.06261"
  doi: null
  s2: null
tags: ["amodal-segmentation", "occlusion", "SAM", "CVPR-2025"]
added: 2026-08-03T09:17:08Z
---

# Segment Anything, Even Occluded

## One-line thesis
A SAM-based decoder can be adapted to predict complete masks under occlusion using large-scale synthetic amodal supervision.

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

> Amodal instance segmentation, which aims to detect and segment both visible and invisible parts of objects in images, plays a crucial role in various applications including autonomous driving, robotic manipulation, and scene understanding. While existing methods require training both front-end detectors and mask decoders jointly, this approach lacks flexibility and fails to leverage the strengths of pre-existing modal detectors. To address this limitation, we propose SAMEO, a novel framework that adapts the Segment Anything Model (SAM) as a versatile mask decoder capable of interfacing with various front-end detectors to enable mask prediction even for partially occluded objects. Acknowledging the constraints of limited amodal segmentation datasets, we introduce Amodal-LVIS, a large-scale synthetic dataset comprising 300K images derived from the modal LVIS and LVVIS datasets. This dataset significantly expands the training data available for amodal segmentation research. Our experimental results demonstrate that our approach, when trained on the newly extended dataset, including Amodal-LVIS, achieves remarkable zero-shot performance on both COCOA-cls and D2SA benchmarks, highlighting its potential for generalization to unseen scenarios.

