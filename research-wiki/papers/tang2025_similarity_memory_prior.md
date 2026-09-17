---
type: paper
node_id: paper:tang2025_similarity_memory_prior
title: "Similarity Memory Prior is All You Need for Medical Image Segmentation"
authors: ["Hao Tang", "Zhiqing Guo", "Liejun Wang", "Chao Liu"]
year: 2025
venue: "ICCV 2025"
external_ids:
  arxiv: "2507.00585"
  doi: null
  s2: null
tags: ["medical-segmentation", "memory-prior", "prototype"]
added: 2026-07-27T04:42:29Z
---

# Similarity Memory Prior is All You Need for Medical Image Segmentation

## One-line thesis
Dynamically learns class-specific similarity prototypes in a memory bank for segmentation.

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

> In recent years, it has been found that "grandmother cells" in the primary visual cortex (V1) of macaques can directly recognize visual input with complex shapes. This inspires us to examine the value of these cells in promoting the research of medical image segmentation. In this paper, we design a Similarity Memory Prior Network (Sim-MPNet) for medical image segmentation. Specifically, we propose a Dynamic Memory Weights-Loss Attention (DMW-LA), which matches and remembers the category features of specific lesions or organs in medical images through the similarity memory prior in the prototype memory bank, thus helping the network to learn subtle texture changes between categories. DMW-LA also dynamically updates the similarity memory prior in reverse through Weight-Loss Dynamic (W-LD) update strategy, effectively assisting the network directly extract category features. In addition, we propose the Double-Similarity Global Internal Enhancement Module (DS-GIM) to deeply explore the internal differences in the feature distribution of input data through cosine similarity and euclidean distance. Extensive experiments on four public datasets show that Sim-MPNet has better segmentation performance than other state-of-the-art methods. Our code is available on https://github.com/vpsg-research/Sim-MPNet.

