---
type: experiment
node_id: exp:error_refiner_unet512_b32_local_test
added: 2026-06-15T12:55:00Z
---

# Experiment: error_refiner_unet512_b32 local test evaluation

## Summary

This is the earliest clearly documented error-refiner milestone available in the local workspace. It established that a learned refiner could outperform zero-shot YOLO+SAM prompting baselines.

## Artifact paths

- metrics: `outputs/ablations/error_refiner_unet512_b32/TSRS_RSNA-Epiphysis/test/metrics.json`
- summary note: `outputs/error_refiner_experiment_summary_2026-05-15.md`

## Mean metrics

- Dice: 0.883870
- IoU: 0.795120
- Precision: 0.867619
- Recall: 0.904690
- Specificity: 0.996141
- Boundary IoU: 0.194766

## Interpretation

This experiment shows a meaningful step up over the project's early zero-shot baselines, but it is not the final performance frontier of the repository.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
