---
type: experiment
node_id: exp:interaction_refiner_server_line
added: 2026-06-15T13:25:00Z
---

# Experiment: interaction-refiner server line

## Summary

This page captures the server-side interaction-refiner branch as a meaningful intermediate step in the anatomy-aware refiner family.

## Launcher

- local launcher: `run_epiphysis_allstack_anatomy_roi_interaction_refiner.sh`
- target mode: `allstack_anatomy_roi_interaction_refiner`
- canonical server workspace: `/home/shenzeyu/workspace/YOLO_SAM_generic_src`

## Role in the project

- this line was stronger than earlier generic refiners
- it helped confirm that simple smoothing is not enough and that interaction-aware refinement can improve contour behavior
- however, it still did not produce a decisive margin over the stronger later keepbone/cutgap family

## Current interpretation

- useful as an internal anchor, not as the final project frontier
- still relevant for bad-case comparison because it represents a different trade-off than the later stage2 structure-aware variants

## Expected artifacts on server

- checkpoint root: `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/error_refiner/TSRS_RSNA-Epiphysis`
- output name: `allstack_anatomy_roi_interaction_refiner_sam_hqsam`

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
