# R201 Paper Main-Table Recommendation

## Recommended Main Table

Use Dice, IoU, Precision, Recall, Boundary IoU, Boundary F1, Surface Dice 2px, HD95, and ASSD.

| System | Evidence | N | dice | iou | precision | recall | boundary_iou | boundary_f1 | surface_dice_2px | hd95_px | assd_px |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R202 nnU-Net 2D baseline | full_mask_recomputed | 81 | 0.923573 | 0.858508 | 0.907984 | 0.941007 | 0.270431 | 0.417773 | 0.713582 | 6.113146 | 2.083621 |
| R110 current best | full_mask_recomputed | 81 | 0.917723 | 0.848570 | 0.913343 | 0.923580 | 0.251896 | 0.394721 | 0.684156 | 7.220072 | 2.419143 |
| R100 patch complement | full_mask_recomputed | 81 | 0.917618 | 0.848394 | 0.912686 | 0.924039 | 0.251759 | 0.394545 | 0.684012 | 7.231737 | 2.423387 |
| R090 patch arbitrator | full_mask_recomputed | 81 | 0.916440 | 0.846313 | 0.906271 | 0.928080 | 0.251366 | 0.394480 | 0.685533 | 6.452675 | 2.276369 |
| ARAA DANet epoch98 | full_mask_recomputed | 81 | 0.911766 | 0.838436 | 0.910177 | 0.914691 | 0.235265 | 0.374350 | 0.677752 | 6.596897 | 2.343013 |
| R143 high-res medical U-Net recipe | full_mask_recomputed | 81 | 0.891375 | 0.804630 | 0.821758 | 0.975703 | 0.161392 | 0.271987 | 0.505058 | 11.686893 | 3.706147 |
| R130 bridge-suppressed source | full_mask_recomputed | 81 | 0.890938 | 0.803897 | 0.855765 | 0.931193 | 0.180672 | 0.301228 | 0.550674 | 8.184853 | 2.912025 |
| R165 filtered reannotation source | full_mask_recomputed | 81 | 0.890303 | 0.802926 | 0.878265 | 0.905050 | 0.184990 | 0.307025 | 0.559593 | 8.041408 | 2.961148 |

## Recommended Diagnostic Table

Keep this separate from the main table and label it as anatomical consistency / failure-mode diagnostics.

| System | Evidence | N | gap_region_fp_rate | component_merge_rate | component_count_mae |
| --- | --- | --- | --- | --- | --- |
| R090 patch arbitrator | full_mask_recomputed | 81 | 0.213006 | 0.074074 | 9.345679 |
| R130 bridge-suppressed source | full_mask_recomputed | 81 | 0.317425 | 0.641975 | 1.543210 |
| R165 filtered reannotation source | full_mask_recomputed | 81 | 0.252102 | 0.641975 | 1.666667 |
| ARAA DANet epoch98 | full_mask_recomputed | 81 | 0.201259 | 0.679012 | 2.086420 |
| R202 nnU-Net 2D baseline | full_mask_recomputed | 81 | 0.218767 | 0.679012 | 1.827160 |
| R100 patch complement | full_mask_recomputed | 81 | 0.202583 | 0.790123 | 2.506173 |
| R110 current best | full_mask_recomputed | 81 | 0.201358 | 0.790123 | 2.506173 |
| R143 high-res medical U-Net recipe | full_mask_recomputed | 81 | 0.450644 | 0.888889 | 3.913580 |
