---
type: experiment
node_id: exp:r015_selector_aic_recall_calibration
title: "R015 selector calibration with AIC recall cue"
date: 2026-06-22
status: succeeded
---

# R015 selector calibration with AIC recall cue

## Setup

- Dataset: `TSRS_RSNA-Epiphysis`
- Primary slice: `TSRS_RSNA-Epiphysis_clean_test_v2`
- Output experiment: `allstack_anatomy_roi_selector_r015_sam_hqsam`
- Launcher: `run_epiphysis_selector_r015.sh`
- Rule file: `outputs/error_refiner/TSRS_RSNA-Epiphysis/selector_r015_rule.json`

## Selected Rule

- base: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam`
- add: `allstack_anatomy_roi_aic_refiner_sam_hqsam`
- add radius: `1`
- max add component: `1000000`
- support minimum: `0`
- close kernel: `1`

## Results

| Slice | Dice | IoU | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| original test | 0.896905 | 0.815969 | 0.884421 | 0.912982 | 0.225330 |
| clean-test-v2 | 0.913497 | 0.841307 | 0.899351 | 0.929412 | 0.239510 |

## Verdict

Succeeded. R015 exceeds the external ARAA clean-test-v2 Dice target `0.911766` and also improves original-test Dice over the internal anchor `0.893694`.

The successful pattern is not a new trained refiner. It is validation-calibrated local recall recovery: start from the precision-tuned keepbone/cutgap mask and add only AIC regions within a one-pixel neighborhood of the base prediction.

## Reproduction

```bash
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
CUDA_VISIBLE_DEVICES=1 bash run_epiphysis_selector_r015.sh
```

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
