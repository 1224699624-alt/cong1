---
type: experiment
node_id: exp:araa_clean_test_v2_checkpoint_sweep
added: 2026-06-15T13:25:00Z
---

# Experiment: ARAA clean-test-v2 checkpoint sweep

## Summary

This page records the server-side checkpoint sweep that established the correct ARAA comparison target under the user's current environment and evaluation protocol.

## Server context

- repo: `/home/shenzeyu/workspace/ARAA-Net`
- purpose: identify the true best checkpoint for `TSRS_RSNA-Epiphysis_clean_test_v2`
- sweep summary: `/home/shenzeyu/workspace/ARAA-Net/outputs/tsrs_epiphysis_clean_test_v2_ckpt_sweep/summary.json`

## Key finding

- the true best checkpoint is `98.pth`
- this corrected earlier ambiguity about which checkpoint should represent ARAA in downstream comparisons

## Best metrics

- Dice: `0.911766`
- IoU: `0.838436`
- Precision: `0.910177`
- Recall: `0.914691`
- Specificity: `0.997176`
- Boundary IoU: `0.235265`

## Best artifact

- metrics: `/home/shenzeyu/workspace/ARAA-Net/outputs/tsrs_epiphysis_clean_test_v2_ckpt_sweep/epoch_98/metrics.json`

## Why this matters

- this sweep fixed the benchmark target that the current refiner line still needs to beat
- it also established that ARAA's remaining edge is mostly on recall and boundary completeness, not on precision

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
