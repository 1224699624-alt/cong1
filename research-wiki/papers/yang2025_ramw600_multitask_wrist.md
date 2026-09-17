---
type: paper
node_id: paper:yang2025_ramw600_multitask_wrist
title: "RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis"
authors: ["Songxiao Yang", "Haolin Wang", "Yao Fu", "Ye Tian", "Tamotsu Kamishima", "Masayuki Ikebe", "Yafei Ou", "Masatoshi Okutomi"]
year: 2025
venue: "arXiv"
external_ids:
  arxiv: "2507.05193"
  doi: null
  s2: null
tags: ["dataset", "wrist-xray", "instance-segmentation", "joint-space"]
added: 2026-07-22T04:21:23Z
---

# RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis

## One-line thesis
公开腕部X光逐骨实例分割数据，狭窄关节间隙和骨重叠可直接验证通用seam先验。

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

> Rheumatoid arthritis (RA) is a common autoimmune disease that has been the focus of research in computer-aided diagnosis (CAD) and disease monitoring. In clinical settings, conventional radiography (CR) is widely used for the screening and evaluation of RA due to its low cost and accessibility. The wrist is a critical region for the diagnosis of RA. However, CAD research in this area remains limited, primarily due to the challenges in acquiring high-quality instance-level annotations. (i) The wrist comprises numerous small bones with narrow joint spaces, complex structures, and frequent overlaps, requiring detailed anatomical knowledge for accurate annotation. (ii) Disease progression in RA often leads to osteophyte, bone erosion (BE), and even bony ankylosis, which alter bone morphology and increase annotation difficulty, necessitating expertise in rheumatology. This work presents a multi-task dataset for wrist bone in CR, including two tasks: (i) wrist bone instance segmentation and (ii) Sharp/van der Heijde (SvdH) BE scoring, which is the first public resource for wrist bone instance segmentation. This dataset comprises 1048 wrist conventional radiographs of 388 patients from six medical centers, with pixel-level instance segmentation annotations for 618 images and SvdH BE scores for 800 images. This dataset can potentially support a wide range of research tasks related to RA, including joint space narrowing (JSN) progression quantification, BE detection, bone deformity evaluation, and osteophyte detection. It may also be applied to other wrist-related tasks, such as carpal bone fracture localization. We hope this dataset will significantly lower the barrier to research on wrist RA and accelerate progress in CAD research within the RA-related domain.

