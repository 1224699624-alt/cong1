---
type: paper
node_id: paper:gao2023_coarsetofine_amodal_segmentation
title: "Coarse-to-Fine Amodal Segmentation with Shape Prior"
authors: ["Jianxiong Gao", "Xuelin Qian", "Yikai Wang", "Tianjun Xiao", "Tong He", "Zheng Zhang", "Yanwei Fu"]
year: 2023
venue: "arXiv"
external_ids:
  arxiv: "2308.16825"
  doi: null
  s2: null
tags: ["amodal-segmentation", "shape-prior", "coarse-to-fine", "ICCV-2023"]
added: 2026-08-03T09:17:10Z
---

# Coarse-to-Fine Amodal Segmentation with Shape Prior

## One-line thesis
A vector-quantized coarse shape prior followed by image-feature refinement recovers occluded object extent more reliably than direct pixel prediction.

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

> Amodal object segmentation is a challenging task that involves segmenting both visible and occluded parts of an object. In this paper, we propose a novel approach, called Coarse-to-Fine Segmentation (C2F-Seg), that addresses this problem by progressively modeling the amodal segmentation. C2F-Seg initially reduces the learning space from the pixel-level image space to the vector-quantized latent space. This enables us to better handle long-range dependencies and learn a coarse-grained amodal segment from visual features and visible segments. However, this latent space lacks detailed information about the object, which makes it difficult to provide a precise segmentation directly. To address this issue, we propose a convolution refine module to inject fine-grained information and provide a more precise amodal object segmentation based on visual features and coarse-predicted segmentation. To help the studies of amodal object segmentation, we create a synthetic amodal dataset, named as MOViD-Amodal (MOViD-A), which can be used for both image and video amodal object segmentation. We extensively evaluate our model on two benchmark datasets: KINS and COCO-A. Our empirical results demonstrate the superiority of C2F-Seg. Moreover, we exhibit the potential of our approach for video amodal object segmentation tasks on FISHBOWL and our proposed MOViD-A. Project page at: http://jianxgao.github.io/C2F-Seg.

