# R201 Baseline Metric Brief

## Scope
- Dataset: `TSRS_RSNA-Epiphysis_clean_test_v2/test`
- Evidence: R201 unified comparison table
- Note: `hd95_px` and `assd_px` are lower-is-better pixel-unit fields.

## Primary Rows
| model | evaluated | Dice | IoU | BIoU | BF1 | SD2 | SD5 | HD95px | ASSDpx | gap FP | merge | count MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| araa_danet_epoch98 | 81/0 missing | 0.911766 | 0.838436 | 0.235265 | 0.374350 | 0.677752 | 0.917918 | 6.596897 | 2.343013 | 0.201259 | 0.679012 | 2.086420 |
| r110_r100_r108_patch_basic | 81/0 missing | 0.917723 | 0.848570 | 0.251896 | 0.394721 | 0.684156 | 0.915014 | 7.220072 | 2.419143 | 0.201358 | 0.790123 | 2.506173 |
| r143_highres_medical_recipe_unet | 81/0 missing | 0.891375 | 0.804630 | 0.161392 | 0.271987 | 0.505058 | 0.835301 | 11.686893 | 3.706147 | 0.450644 | 0.888889 | 3.913580 |
| r202_nnunet2d | 81/0 missing | 0.923573 | 0.858508 | 0.270431 | 0.417773 | 0.713582 | 0.930538 | 6.113146 | 2.083621 | 0.218767 | 0.679012 | 1.827160 |

## Metric Leaders Among Complete R201 Rows
- `dice` (higher): `r202_nnunet2d` = `0.923573`
- `iou` (higher): `r202_nnunet2d` = `0.858508`
- `boundary_iou` (higher): `r202_nnunet2d` = `0.270431`
- `boundary_f1` (higher): `r202_nnunet2d` = `0.417773`
- `surface_dice_2px` (higher): `r202_nnunet2d` = `0.713582`
- `surface_dice_5px` (higher): `r202_nnunet2d` = `0.930538`
- `hd95_px` (lower): `r202_nnunet2d` = `6.113146`
- `assd_px` (lower): `r202_nnunet2d` = `2.083621`
- `gap_region_fp_rate` (lower): `araa_danet_epoch98` = `0.201259`
- `component_merge_rate` (lower): `r090_online_patch_disagreement_arbitrator` = `0.074074`
- `component_count_mae` (lower): `r130_dinov3_bridge_suppressed_instance_sep` = `1.543210`

## Current Bottleneck
- R110 improves Dice/IoU/Boundary IoU over ARAA, but has worse component merge and component count MAE.
- nnU-Net is a risk-audit upper baseline for overlap/boundary, but has worse gap-region FP than ARAA/R110.
- New R110-based variants should be judged by anatomy diagnostics first: gap FP, component merge, component count MAE, and boundary/surface metrics, while keeping Dice/IoU competitive.
