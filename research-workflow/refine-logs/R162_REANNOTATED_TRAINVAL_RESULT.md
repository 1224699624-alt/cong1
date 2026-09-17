# R162 Reannotated Train/Val Result

## Status

DONE_NEGATIVE.

R162 trained the guarded DINOv3 instance-separation source model on `TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1`, where train/val come from the reannotated flat-output variant and test is copied only from the original test split. The success evaluation used only `TSRS_RSNA-Epiphysis_clean_test_v2/test`.

## Result

- Best validation epoch: `21`
- Best validation Dice: `0.8124495754767102`
- Clean-test-v2 evaluated images: `81`
- Clean-test-v2 Dice: `0.8726263750014701`
- Clean-test-v2 Precision: `0.8581884012740948`
- Clean-test-v2 Recall: `0.8907767291374148`
- Clean-test-v2 Boundary IoU: `0.1608712562661216`
- Clean-test-v2 false-bridge flag: `0.32098765432098764`
- Current valid best R110 Dice: `0.9177231563529792`
- Target Dice: `0.9317660066557425`
- Delta vs R110: `-0.0450967813515091`
- Gap to target: `0.0591396316542724`

## Notes

The final clean-test-v2 metric is valid and leakage-safe. The original-test control JSON is empty because the R162 launcher passed `--raw-root data/raw_variants` together with `--control-dataset TSRS_RSNA-Epiphysis`; that control dataset is not present under `data/raw_variants`. This does not affect the clean-test-v2 success metric, but future launchers should pass an explicit control root or skip the control output when the training raw root differs from `data/raw`.

Training became numerically unstable after the best epoch: later epochs wrote `nan` train loss and zero validation Dice, but the saved checkpoint came from epoch 21 before the collapse.

## Decision

Do not continue this simple reannotated train/val retraining branch. The new labels alone did not improve over R110 and are substantially below the target. The next ARIS step should be a decision branch, not another blind R162-style rerun: either inspect why reannotated labels degrade validation transfer, or require a materially stronger architecture/environment change.
