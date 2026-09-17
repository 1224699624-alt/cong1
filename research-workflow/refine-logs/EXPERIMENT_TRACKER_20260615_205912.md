# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | freeze external anchor | ARAA `epoch_98` | `TSRS_RSNA-Epiphysis_clean_test_v2` | Dice, IoU, Precision, Recall, Boundary IoU | MUST | TODO | read remote metrics and record exact path |
| R002 | M0 | freeze current internal best on primary slice | `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam` | `TSRS_RSNA-Epiphysis_clean_test_v2` | Dice, IoU, Precision, Recall, Boundary IoU | MUST | TODO | confirm clean-test-v2 metrics and summary path |
| R003 | M0 | freeze current internal best on control slice | `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam` | `TSRS_RSNA-Epiphysis` test | Dice, IoU, Precision, Recall, Boundary IoU | MUST | TODO | record original test metrics path |
| R004 | M1 | full candidate train + threshold sweep | `allstack_anatomy_roi_aic_refiner_sam_hqsam` | train/val | val Dice-led threshold selection | MUST | TODO | use `run_epiphysis_allstack_anatomy_roi_aic_refiner.sh` |
| R005 | M1 | candidate evaluation on original test | `allstack_anatomy_roi_aic_refiner_sam_hqsam` | `TSRS_RSNA-Epiphysis` test | Dice, IoU, Precision, Recall, Specificity, Boundary IoU | MUST | TODO | expected under `outputs/ablations/.../test/metrics.json` |
| R006 | M1 | candidate evaluation on primary slice | `allstack_anatomy_roi_aic_refiner_sam_hqsam` | `TSRS_RSNA-Epiphysis_clean_test_v2` | Dice, IoU, Precision, Recall, Specificity, Boundary IoU | MUST | TODO | run with `REFINER_EXP=... bash run_epiphysis_clean_test_keepbone_cutgap_refiner_v2.sh` |
| R007 | M3 | anatomical hard-case validation | baseline vs current best vs candidate | `2982/1475/3040/1433` | qualitative gap / thin-epiphysis preservation | MUST | TODO | export comparison figures and short judgment |
| R008 | M4 | filtered-data control | keepbone-cutgap filtered trainval v1 | original test + clean-test-v2 if available | same core metrics | NICE | TODO | run only if R006 is positive or near-positive |
| R009 | M4 | filtered-data control | keepbone-cutgap badlabel filtered v1 | original test + clean-test-v2 if available | same core metrics | NICE | TODO | keep isolated outputs and separate reporting |
