---
type: paper
node_id: paper:wang2024_blsgan_deep_layer
title: "BLS-GAN: A Deep Layer Separation Framework for Eliminating Bone Overlap in Conventional Radiographs"
authors: ["Haolin Wang", "Yafei Ou", "Prasoon Ambalathankandy", "Gen Ota", "Pengyu Dai", "Masayuki Ikebe", "Kenji Suzuki", "Tamotsu Kamishima"]
year: 2025
venue: "AAAI 2025"
external_ids:
  arxiv: "2409.07304"
  doi: "10.1609/aaai.v39i7.32826"
  s2: null
tags: ["bone-overlap", "x-ray", "layer-separation", "AAAI-2025"]
added: 2026-08-03T04:49:05Z
---

# BLS-GAN: A Deep Layer Separation Framework for Eliminating Bone Overlap in Conventional Radiographs

## One-line thesis
Physics-informed layer separation can disentangle superposed bone texture in conventional radiographs and motivates overlap-aware anatomical priors.

## Problem / Gap
Conventional radiographs superpose multiple bone layers, so anatomy and texture in an overlap cannot be assessed independently by either clinicians or downstream algorithms.

## Method
BLS-GAN treats overlap as a radiographic layer-separation problem rather than an ordinary mutually exclusive segmentation problem. It uses synthetic pretraining and a reconstruction module derived from conventional-radiography image formation to constrain the generated bone layers to reconstruct the observed projection.

## Key Results
The generated layers passed the reported visual Turing evaluation and improved downstream analysis. The paper establishes the feasibility of bone-layer separation, but it does not directly report wrist instance-segmentation masks as its primary output.

## Assumptions
The observed projection can be decomposed into plausible constituent bone layers under the learned data distribution and the reconstruction model.

## Limitations / Failure Modes
Layer images are not the same as amodal instance masks. GAN outputs may be plausible without being anatomically exact, and the method relies on synthetic pretraining and a task-specific image-formation model.

## Reusable Ingredients
Physics-informed reconstruction consistency, synthetic overlap generation, and a layer-separation teacher whose features or outputs can guide a downstream multi-label segmenter.

## Open Questions
Whether layer-separation supervision improves RAM-W600 per-bone amodal masks without hallucinating incorrect anatomy.

## Claims
_TODO._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
This is the most directly relevant recent peer-reviewed work on bone overlap in conventional radiographs. It supports adding projection-consistency or layer-separation supervision, but should not be mistaken for a ready-made segmentation prior.

## Local code audit and RAM test

The public code confirms that the generator consumes both the radiograph and bone masks, applies the masks to generated layers, and learns a scalar overlap correction in the reconstructor. R326 generalized the idea to fourteen predicted RAM bone layers and an inference-time post-logit adapter. The complete R326 model improved all tracked metrics over the frozen RAM baseline, but a capacity-matched segmentation-only adapter remained better on Overlap DSC/IoU/NSD. This supports the adapter architecture but does not yet isolate a positive overlap-recovery effect from the projection prior.

## Abstract (original)

> Conventional radiography is the widely used imaging technology in diagnosing, monitoring, and prognosticating musculoskeletal (MSK) diseases because of its easy availability, versatility, and cost-effectiveness. In conventional radiographs, bone overlaps are prevalent, and can impede the accurate assessment of bone characteristics by radiologists or algorithms, posing significant challenges to conventional and computer-aided diagnoses. This work initiated the study of a challenging scenario - bone layer separation in conventional radiographs, in which separate overlapped bone regions enable the independent assessment of the bone characteristics of each bone layer and lay the groundwork for MSK disease diagnosis and its automation. This work proposed a Bone Layer Separation GAN (BLS-GAN) framework that can produce high-quality bone layer images with reasonable bone characteristics and texture. This framework introduced a reconstructor based on conventional radiography imaging principles, which achieved efficient reconstruction and mitigates the recurrent calculations and training instability issues caused by soft tissue in the overlapped regions. Additionally, pre-training with synthetic images was implemented to enhance the stability of both the training process and the results. The generated images passed the visual Turing test, and improved performance in downstream tasks. This work affirms the feasibility of extracting bone layer images from conventional radiographs, which holds promise for leveraging bone layer separation technology to facilitate more comprehensive analytical research in MSK diagnosis, monitoring, and prognosis. Code and dataset: https://github.com/pokeblow/BLS-GAN.git.

