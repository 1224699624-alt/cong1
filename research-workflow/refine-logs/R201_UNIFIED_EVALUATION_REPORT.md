# R201 Unified Evaluation Report

## Regression Status

- R200 anchor regression passed: `True`
- R110 remains above ARAA on Dice/IoU/Boundary IoU, while its component/gap diagnostics remain worse. This preserves the earlier interpretation: R110 is a strong overlap/boundary anchor but has not solved bone-gap adhesion.

## Main Table Candidates

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

## Anatomy Diagnostic Ranking

| System | Evidence | N | component_merge_rate | component_count_mae | gap_region_fp_rate | dice | iou |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R090 patch arbitrator | full_mask_recomputed | 81 | 0.074074 | 9.345679 | 0.213006 | 0.916440 | 0.846313 |
| R130 bridge-suppressed source | full_mask_recomputed | 81 | 0.641975 | 1.543210 | 0.317425 | 0.890938 | 0.803897 |
| R165 filtered reannotation source | full_mask_recomputed | 81 | 0.641975 | 1.666667 | 0.252102 | 0.890303 | 0.802926 |
| ARAA DANet epoch98 | full_mask_recomputed | 81 | 0.679012 | 2.086420 | 0.201259 | 0.911766 | 0.838436 |
| R202 nnU-Net 2D baseline | full_mask_recomputed | 81 | 0.679012 | 1.827160 | 0.218767 | 0.923573 | 0.858508 |
| R100 patch complement | full_mask_recomputed | 81 | 0.790123 | 2.506173 | 0.202583 | 0.917618 | 0.848394 |
| R110 current best | full_mask_recomputed | 81 | 0.790123 | 2.506173 | 0.201358 | 0.917723 | 0.848570 |
| R143 high-res medical U-Net recipe | full_mask_recomputed | 81 | 0.888889 | 3.913580 | 0.450644 | 0.891375 | 0.804630 |

## Hard-Case Visualization Queue

Top R110-vs-ARAA disagreement cases for qualitative panels are saved in `outputs\analysis\r201_unified_eval\r201_r110_vs_araa_hard_case_queue.csv`. These are the first cases to inspect when showing adhesion/merge failures.

## Current Claim Audit

- Supported now: R110 improves overlap and boundary metrics over ARAA.
- Not supported yet: R110 solves bone-gap adhesion/component merging over ARAA.
- Required next evidence: a method or postprocessor that keeps Dice/IoU at least at ARAA and ideally R110 level while lowering component merge rate, component count MAE, and gap-region FP rate.

## Strong Baseline Queue

- First priority: nnU-Net trained only on train/val, with clean-test-v2 inference after checkpoint/threshold lock.
- Secondary baselines after nnU-Net: Swin UNETR or MedNeXt if compute and conversion costs are acceptable; MaskDINO binary-union remains experimental because earlier train/val gates were not yet a clean-test-v2-ready baseline.
