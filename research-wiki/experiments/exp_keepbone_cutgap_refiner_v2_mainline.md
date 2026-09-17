---
type: experiment
node_id: exp:keepbone_cutgap_refiner_v2_mainline
added: 2026-06-15T13:25:00Z
---

# Experiment: keepbone-cutgap-refiner-v2 mainline

## Summary

This page records the current strongest YOLO+SAM internal refiner line that nearly matches ARAA on `clean-test-v2`.

## Launcher and mode

- local launcher: `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v2.sh`
- clean-test evaluation launcher: `run_epiphysis_clean_test_keepbone_cutgap_refiner_v2.sh`
- target mode: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2`

## Main design idea

- prediction-aware keepbone/cutgap stage2 correction
- one side preserves thin positive bone structures
- one side protects narrow negative gaps between adjacent bones
- the method tries to improve separation without falling back to blunt closing-style cleanup

## Best recorded clean-test-v2 metrics

- Dice: `0.910029`
- IoU: `0.835434`
- Precision: `0.913101`
- Recall: `0.908201`
- Specificity: `0.997178`
- Boundary IoU: `0.228489`

## Best known artifact on server

- metrics: `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/ablations_variants/allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam/TSRS_RSNA-Epiphysis_clean_test_v2/test/metrics.json`

## Comparison to ARAA

- still slightly below ARAA overall on clean-test-v2
- somewhat more conservative than ARAA
- precision is slightly better, but recall and boundary completeness remain lower

## Why this page matters

- this is the current internal score to beat before any new direction can be promoted
- it is the natural bridge between earlier interaction/two-stage lines and the next ARAA-targeting candidate

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
