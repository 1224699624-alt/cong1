---
type: paper
node_id: paper:ozguroglu2024_pix2gestalt
title: "pix2gestalt: Amodal Segmentation by Synthesizing Wholes"
authors: ["Ege Ozguroglu", "Ruoshi Liu", "Dídac Surís", "Dian Chen", "Achal Dave", "Pavel Tokmakov", "Carl Vondrick"]
year: 2024
venue: "CVPR 2024"
external_ids:
  arxiv: "2401.14398"
  doi: null
  s2: null
tags: ["amodal-segmentation", "occlusion", "diffusion-prior", "CVPR-2024"]
added: 2026-08-03T12:00:00Z
---

# pix2gestalt: Amodal Segmentation by Synthesizing Wholes

## One-line thesis
A pretrained generative model can provide an implicit complete-object prior by synthesizing the unoccluded whole before extracting its amodal mask.

## Problem / Gap
Occluded pixels provide no direct appearance evidence, making complete-instance masks hard to infer using an ordinary modal segmenter.

## Method
The method conditions a diffusion model on the observed image and target object to synthesize a plausible unoccluded whole, from which an amodal segmentation is derived.

## Key Results
The official CVPR paper reports strong zero-shot results on Amodal COCO and Amodal Berkeley benchmarks.

## Assumptions
The generative model contains a useful category-level complete-shape prior and can preserve observed object identity while completing hidden regions.

## Limitations / Failure Modes
Plausible natural-image synthesis can violate physical or instance-specific anatomy. Direct RGB generation is therefore unsafe as ground truth for wrist radiographs without projection and visible-region consistency constraints.

## Reusable Ingredients
Inference-time complete-shape generation, explicit separation of observed and hidden regions, and consistency checking between generated whole and visible evidence.

## Relevance to This Project
It supports a learned completion module that remains active during inference. For RAM-W600, mask- or latent-space completion is safer than synthesizing X-ray pixels, and must use per-bone multi-label output.

## Primary Source

- https://openaccess.thecvf.com/content/CVPR2024/html/Ozguroglu_pix2gestalt_Amodal_Segmentation_by_Synthesizing_Wholes_CVPR_2024_paper.html
