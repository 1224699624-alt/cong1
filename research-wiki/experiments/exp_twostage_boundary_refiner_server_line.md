---
type: experiment
node_id: exp:twostage_boundary_refiner_server_line
added: 2026-06-15T13:25:00Z
---

# Experiment: two-stage boundary-refiner server line

## Summary

This page records the two-stage boundary correction branch that was introduced to refine only a narrow boundary band rather than globally smoothing the mask.

## Launcher

- local launcher: `run_epiphysis_allstack_anatomy_roi_twostage_boundary_refiner.sh`
- target mode: `allstack_anatomy_roi_twostage_boundary_refiner`
- canonical server workspace: `/home/shenzeyu/workspace/YOLO_SAM_generic_src`

## Research intent

- coarse segmentation first
- then boundary-band correction at a more structure-sensitive stage
- designed to protect thin epiphysis regions and bone gaps better than naive morphological cleanup

## Current interpretation

- this line was an important conceptual step toward structure-sensitive stage2 correction
- it helped narrow down the project bottleneck, but later keepbone/cutgap variants became the stronger mainline

## Expected artifacts on server

- checkpoint root: `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/error_refiner/TSRS_RSNA-Epiphysis`
- output name: `allstack_anatomy_roi_twostage_boundary_refiner_sam_hqsam`

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
