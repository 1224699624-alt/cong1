---
type: paper
node_id: paper:tran2024_shapeformer_shape_prior
title: "ShapeFormer: Shape Prior Visible-to-Amodal Transformer-based Amodal Instance Segmentation"
authors: ["Minh Tran", "Winston Bounsavy", "Khoa Vo", "Anh Nguyen", "Tri Nguyen", "Ngan Le"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2403.11376"
  doi: null
  s2: null
tags: ["amodal-segmentation", "shape-prior", "occlusion"]
added: 2026-08-03T04:49:17Z
---

# ShapeFormer: Shape Prior Visible-to-Amodal Transformer-based Amodal Instance Segmentation

## One-line thesis
A decoupled visible-to-amodal architecture and retrieved shape priors support recovery of hidden object extent without contaminating visible-mask features.

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

> Amodal Instance Segmentation (AIS) presents a challenging task as it involves predicting both visible and occluded parts of objects within images. Existing AIS methods rely on a bidirectional approach, encompassing both the transition from amodal features to visible features (amodal-to-visible) and from visible features to amodal features (visible-to-amodal). Our observation shows that the utilization of amodal features through the amodal-to-visible can confuse the visible features due to the extra information of occluded/hidden segments not presented in visible display. Consequently, this compromised quality of visible features during the subsequent visible-to-amodal transition. To tackle this issue, we introduce ShapeFormer, a decoupled Transformer-based model with a visible-to-amodal transition. It facilitates the explicit relationship between output segmentations and avoids the need for amodal-to-visible transitions. ShapeFormer comprises three key modules: (i) Visible-Occluding Mask Head for predicting visible segmentation with occlusion awareness, (ii) Shape-Prior Amodal Mask Head for predicting amodal and occluded masks, and (iii) Category-Specific Shape Prior Retriever aims to provide shape prior knowledge. Comprehensive experiments and extensive ablation studies across various AIS benchmarks demonstrate the effectiveness of our ShapeFormer. The code is available at: \url{https://github.com/UARK-AICV/ShapeFormer}

