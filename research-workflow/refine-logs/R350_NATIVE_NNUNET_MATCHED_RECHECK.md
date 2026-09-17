# R350 Native nnU-Net Matched Recheck

## Purpose

Recompute the TSRS mature native nnU-Net baseline because the historical
HD95 (`54.3414 px`) and ASSD (`14.1738 px`) appear inconsistent with the much
smaller R317/R350 values.

## Frozen protocol

- Dataset: `TSRS_RSNA-Epiphysis` only.
- Split: 96-image `original-val` only.
- Checkpoint: untouched `r202_two_channel_zero_prior.pth`, SHA256 recorded by
  R317 initialization as
  `55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b`.
- Input: the exact `imagesValPrior` directory used by R350. The native
  checkpoint's second-channel weights are zero, so the additional channel
  cannot contribute to its prediction.
- Predictor: nnU-Net v2, Dataset204, 2d, fold 0, same conversion command as
  R350.
- Evaluator: `scripts/evaluate_r201_single_split.py`, exact same GT directory,
  expected count, and defaults as R350.
- No training, threshold search, model selection, or mask editing.
- `clean-test-v2` and `TSRS_RSNA-Articular-Surface` remain unused.

## Isolated outputs

- Masks: `outputs/nnunet/r350_native_recheck/masks/`
- R201 JSON: `outputs/analysis/r350_native_nnunet_matched_original_val_r201.json`
- Log: `outputs/bridge_logs/r350_native_nnunet_matched_recheck.log`

## Result

The matched rerun completed on all 96 cases. The historical baseline values
HD95 `54.3414 px` and ASSD `14.1738 px` are not valid for direct R350
comparison. The matched native values are HD95 `16.963342 px` and ASSD
`4.381480 px`.

Against this corrected native baseline, R350 changes Dice by `-0.00001754`
and IoU by `-0.00003342` (practically tied but strictly lower), while improving
Boundary IoU, Boundary F1, Surface Dice 2/5 px, HD95, ASSD, Gap FP, merge rate,
and component-count MAE. Therefore R350 improves the targeted structural
metrics but does not pass the project's strict Dice/IoU-above-baseline gate.
