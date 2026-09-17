# R235 All-Anchor Coverage Result

## ARIS Status
- Stage: original-val full-val mask-level development
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: yes, isolated output only
- Decision: `R235 = coverage_improved_safe_positive; still_not_R201_ready`

## Purpose
R234 showed that R233 full-val HGB was safe but only evaluated `70` images,
because the application list came from the R224 candidate CSV. The remote
workspace has `96` R110 original-val anchors. R235 tests the same R233 merged
candidate editor with all available anchor masks as the application set.

Training still uses R224 oracle seam samples from original val. Only the
application/source list changed. Clean-test-v2 was not used.

## Code Change
Updated `scripts/run_r233_merged_seam_prob_mask_editor.py`:

- added `--source-mode {source-csv,all-anchors}`;
- default remains `source-csv`, preserving previous R233 behavior;
- `--source-mode all-anchors` enumerates masks from
  `outputs/ablations_variants/<anchor-exp>/<dataset>/<split>/masks`.

## Outputs
- Summary: `outputs/analysis/r235_merged_seam_prob_allanchors_hgb_summary.json`
- Per-image CSV: `outputs/analysis/r235_merged_seam_prob_allanchors_hgb_per_image.csv`
- Candidate CSV: `outputs/analysis/r235_merged_seam_prob_allanchors_hgb_candidates.csv`
- Mask dir: `outputs/ablations_variants/r235_merged_seam_prob_allanchors_hgb/TSRS_RSNA-Epiphysis/val/masks`
- Log: `outputs/bridge_logs/r235_merged_seam_prob_allanchors_hgb.log`
- Visual audit: `outputs/analysis/r235_merged_seam_prob_allanchors_hgb_visual_audit/index.html`

## Key Settings
- `--source-mode all-anchors`
- `--limit-source 0`
- `--limit-train-rows 0`
- `--corridor-radii 1,2`
- `--action-fracs 0.25,0.35,0.50`
- `--ridge-prob-thresholds 0.75,0.85`
- `--ridge-action-fracs 0.20,0.30`
- `--ridge-dilate-radii 0`
- `--recall-tol 0`

## Result
Mean over `96` original-val anchor images:

- edited images: `15`
- Dice delta: `+0.000037`
- IoU delta: `+0.000058`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000135`
- Boundary F1 delta: `+0.000182`
- gap-region FP rate delta: `-0.000143`
- component count MAE delta: `+0.000000`
- mean cut pixels: `3.187500`
- candidate rows: `1089`
- accepted candidates: `15`

Accepted family split:

- probability ridge: `9`
- corridor: `6`

Accepted cases:

| Image | Family | Cut px | Dice Δ | BIoU Δ | BF1 Δ | Gap FP Δ | Component MAE Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10520.png` | corridor | 9 | +0.000145 | +0.000430 | +0.000545 | -0.000499 | +0.000000 |
| `11852.png` | ridge | 31 | +0.000253 | +0.000740 | +0.000938 | -0.001086 | +0.000000 |
| `12827.png` | corridor | 5 | +0.000031 | +0.000191 | +0.000251 | -0.000183 | +0.000000 |
| `13436.png` | ridge | 86 | +0.000630 | +0.000959 | +0.001283 | -0.002550 | +0.000000 |
| `13441.png` | ridge | 7 | +0.000149 | +0.000329 | +0.000377 | -0.000464 | +0.000000 |
| `13517.png` | ridge | 17 | +0.000506 | +0.000200 | +0.000302 | -0.001506 | +0.000000 |
| `13867.png` | corridor | 8 | +0.000073 | +0.000010 | +0.000015 | -0.000341 | +0.000000 |
| `14041.png` | ridge | 41 | +0.000283 | +0.001176 | +0.001403 | -0.001611 | +0.000000 |
| `15446.png` | corridor | 20 | +0.000222 | +0.001815 | +0.002443 | -0.000940 | +0.000000 |
| `1620.png` | corridor | 19 | +0.000099 | +0.000744 | +0.000932 | -0.000592 | +0.000000 |
| `3249.png` | ridge | 24 | +0.000083 | +0.000416 | +0.000627 | -0.000673 | +0.000000 |
| `3602.png` | corridor | 4 | +0.000037 | +0.000178 | +0.000212 | -0.000179 | +0.000000 |
| `5882.png` | ridge | 12 | +0.000536 | +0.002064 | +0.003029 | -0.001478 | +0.000000 |
| `6503.png` | ridge | 14 | +0.000333 | +0.002518 | +0.003720 | -0.001073 | +0.000000 |
| `6698.png` | ridge | 9 | +0.000163 | +0.001188 | +0.001380 | -0.000553 | +0.000000 |

All accepted cases have:

- `cut_gt_fg_frac = 0.0`
- `cut_gt_gap_frac = 1.0`
- `delta_recall = 0.0`

## Comparison To R234
R234 R233 HGB source-list full-val:

- evaluated: `70`
- edited: `9`
- Boundary IoU delta: `+0.000089`
- Boundary F1 delta: `+0.000116`
- gap FP delta: `-0.000116`
- component count MAE delta: `+0.000000`

R235 all-anchor full-val:

- evaluated: `96`
- edited: `15`
- Boundary IoU delta: `+0.000135`
- Boundary F1 delta: `+0.000182`
- gap FP delta: `-0.000143`
- component count MAE delta: `+0.000000`

R235 adds six accepted cases beyond R234:

- `10520.png`
- `13441.png`
- `13517.png`
- `5882.png`
- `6503.png`
- `6698.png`

## Visual Audit
Generated `15` panels:

- `outputs/analysis/r235_merged_seam_prob_allanchors_hgb_visual_audit/index.html`

Inspected representative panels:

- `10520.png`: small corridor cut near adjacent-bone seam; no visible漏分.
- `13436.png`: large but narrow gap cut; still lies in inter-bone background.
- `5882.png`: removes local boundary/gap FP around a small structure; no
  obvious foreground deletion.
- `6503.png`: similar boundary FP cleanup, visually safe in crop.

The visual audit supports the safety of accepted cuts, but most examples are
small local corrections rather than large anatomical separation events.

## Interpretation
R235 confirms that all-anchor coverage improves the R233 route:

- application coverage increased from `70` to `96` images;
- edited-image count increased from `9` to `15`;
- Dice/IoU remained positive;
- Recall stayed unchanged;
- Boundary IoU, Boundary F1, and gap FP moved in the intended direction;
- accepted cuts remain pure gap/background under development diagnostics.

But it still does not meet the project-level target of clear anatomical
consistency improvement:

- effect size is small;
- component count MAE did not improve;
- many candidates are still rejected for `low_mean_prob` or `recall_drop`;
- clean-test-v2 remains locked and should not be used yet.

## Decision
R235 is the best safe full original-val development checkpoint so far, but not
ready for R201 clean-test-v2 promotion.

Next recommended step:

1. keep R235 as the current safe checkpoint;
2. run original-val-only threshold/candidate coverage exploration with strict
   final safety gates unchanged;
3. prioritize increasing safe accepted cases and finding component-merge fixes,
   not simply shaving tiny FP pixels.
