# R213 Validation Gate Audit

Decision: `do_not_promote`

This is validation-only evidence. It is not clean-test-v2 evidence.

## Metrics

| Metric | Value |
| --- | ---: |
| `delta_dice` | `-9.589849103708303e-08` |
| `delta_iou` | `-1.5904192795720973e-07` |
| `delta_boundary_iou` | `2.4803350155185263e-06` |
| `delta_boundary_f1` | `3.3434975659334585e-06` |
| `delta_surface_dice_2px` | `6.466232106558621e-06` |
| `delta_surface_dice_5px` | `-2.332787180074162e-06` |
| `delta_recall` | `-3.122142039210333e-06` |
| `delta_hd95_px` | `0.0` |
| `delta_assd_px` | `-5.815965821859281e-06` |
| `delta_gap_region_fp_rate` | `-7.344646046817734e-06` |
| `delta_component_merge_rate` | `0.0` |
| `delta_component_count_mae` | `0.0` |
| `delta_instance_merge_count` | `0.0` |
| `delta_pred_component_count` | `0.0` |
| `accepted` | `0.010416666666666666` |

## Checks

| Check | Pass |
| --- | ---: |
| `summary_gate_pass` | `True` |
| `accepted_rate_meaningful` | `False` |
| `dice_nonworse` | `True` |
| `iou_nonworse` | `True` |
| `recall_nonworse` | `True` |
| `boundary_iou_meaningful` | `False` |
| `boundary_f1_meaningful` | `False` |
| `surface2_meaningful` | `False` |
| `no_risk_image_accepted` | `True` |
| `gap_fp_improves` | `True` |
| `component_merge_nonworse` | `True` |
| `component_count_nonworse` | `True` |
| `pred_component_count_nonworse` | `True` |

## Accepted Images

- `15446.png`

## Risk Hits

- none
