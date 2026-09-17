---
type: paper
node_id: paper:zhan2023_amodal_ground_truth
title: "Amodal Ground Truth and Completion in the Wild"
authors: ["Guanqi Zhan", "Chuanxia Zheng", "Weidi Xie", "Andrew Zisserman"]
year: 2024
venue: "CVPR 2024"
external_ids:
  arxiv: "2312.17247"
  doi: null
  s2: null
tags: ["amodal-segmentation", "occlusion", "shape-prior", "CVPR-2024"]
added: 2026-08-03T09:17:07Z
---

# Amodal Ground Truth and Completion in the Wild

## One-line thesis
3D-derived authentic amodal ground truth and occluder-aware completion provide a principled route to learning hidden object extent.

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

> This paper studies amodal image segmentation: predicting entire object segmentation masks including both visible and invisible (occluded) parts. In previous work, the amodal segmentation ground truth on real images is usually predicted by manual annotaton and thus is subjective. In contrast, we use 3D data to establish an automatic pipeline to determine authentic ground truth amodal masks for partially occluded objects in real images. This pipeline is used to construct an amodal completion evaluation benchmark, MP3D-Amodal, consisting of a variety of object categories and labels. To better handle the amodal completion task in the wild, we explore two architecture variants: a two-stage model that first infers the occluder, followed by amodal mask completion; and a one-stage model that exploits the representation power of Stable Diffusion for amodal segmentation across many categories. Without bells and whistles, our method achieves a new state-of-the-art performance on Amodal segmentation datasets that cover a large variety of objects, including COCOA and our new MP3D-Amodal dataset. The dataset, model, and code are available at https://www.robots.ox.ac.uk/~vgg/research/amodal/.

