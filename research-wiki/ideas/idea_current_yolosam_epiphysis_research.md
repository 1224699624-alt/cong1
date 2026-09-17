---
type: idea
node_id: idea:current_yolosam_epiphysis_research
title: "Current YOLO+SAM epiphysis refinement research line"
status: active
added: 2026-06-15T12:55:00Z
---

# Current YOLO+SAM Epiphysis Refinement Research Line

## Summary

The active research line in this project aims to improve epiphysis segmentation quality in the YOLO-guided SAM pipeline, with special focus on thin epiphysis preservation, boundary fidelity, and separating adjacent bone gaps.

## Current understanding

- Earlier generic error-refiner models improved over zero-shot baselines.
- Later stronger variants narrowed the gap to ARAA but still struggle to clearly surpass it.
- Label quality and evaluation protocol matter enough that `clean-test-v2` became a necessary secondary reporting slice.

## Main failure modes

- under-segmentation of thin epiphysis regions
- insufficient preservation of narrow inter-bone gaps
- trade-off between smooth edges and small-structure recall
- sensitivity to suspicious annotations

## Current tactical goals

- preserve thin positive anatomy without merging neighboring structures
- improve boundary IoU and Dice together
- separate method gain from label-noise gain
- maintain isolated experiment folders and dataset variants

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
