---
type: experiment
node_id: exp:clean_test_v2_araa_vs_keepbone_cutgap_refiner_v2
added: 2026-06-15T12:55:00Z
---

# Experiment: clean-test-v2 ARAA vs keepbone-cutgap-refiner-v2 comparison

## Summary

This page records a manually curated comparison that has already been verified during project work, even though the full metrics artifacts currently live primarily on the remote server rather than in the local workspace.

## Evaluation slice

- dataset: `TSRS_RSNA-Epiphysis_clean_test_v2`
- purpose: remove a small set of suspicious test annotations and estimate label-noise impact separately from method gain

## Recorded metrics

### ARAA best checkpoint on clean-test-v2

- checkpoint: `98.pth`
- Dice: `0.911766`
- IoU: `0.838436`
- Precision: `0.910177`
- Recall: `0.914691`
- Specificity: `0.997176`
- Boundary IoU: `0.235265`

### Current refiner: allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam

- Dice: `0.910029`
- IoU: `0.835434`
- Precision: `0.913101`
- Recall: `0.908201`
- Specificity: `0.997178`
- Boundary IoU: `0.228489`

## Comparison summary

- ARAA is slightly stronger overall on this slice.
- The strongest visible advantage is higher recall and somewhat better boundary IoU.
- The current refiner is slightly more conservative, with somewhat better precision but lower recall.
- This comparison became an important anchor for later research decisions.

## Known qualitative focus cases

The project repeatedly used difficult samples such as `2982`, `1475`, `3040`, and `1433` to inspect whether methods preserve thin epiphysis structure and narrow bone gaps.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
