---
type: paper
node_id: paper:tran2024_amodal_instance_segmentation
title: "Amodal Instance Segmentation with Diffusion Shape Prior Estimation"
authors: ["Minh Tran", "Khoa Vo", "Tri Nguyen", "Ngan Le"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2409.18256"
  doi: null
  s2: null
tags: ["amodal-segmentation", "diffusion", "shape-prior", "ACCV-2024"]
added: 2026-08-03T09:17:11Z
---

# Amodal Instance Segmentation with Diffusion Shape Prior Estimation

## One-line thesis
A conditioned diffusion shape-prior estimator uses visible masks, occluder masks and category information to refine amodal segmentation.

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

> Amodal Instance Segmentation (AIS) presents an intriguing challenge, including the segmentation prediction of both visible and occluded parts of objects within images. Previous methods have often relied on shape prior information gleaned from training data to enhance amodal segmentation. However, these approaches are susceptible to overfitting and disregard object category details. Recent advancements highlight the potential of conditioned diffusion models, pretrained on extensive datasets, to generate images from latent space. Drawing inspiration from this, we propose AISDiff with a Diffusion Shape Prior Estimation (DiffSP) module. AISDiff begins with the prediction of the visible segmentation mask and object category, alongside occlusion-aware processing through the prediction of occluding masks. Subsequently, these elements are inputted into our DiffSP module to infer the shape prior of the object. DiffSP utilizes conditioned diffusion models pretrained on extensive datasets to extract rich visual features for shape prior estimation. Additionally, we introduce the Shape Prior Amodal Predictor, which utilizes attention-based feature maps from the shape prior to refine amodal segmentation. Experiments across various AIS benchmarks demonstrate the effectiveness of our AISDiff.

