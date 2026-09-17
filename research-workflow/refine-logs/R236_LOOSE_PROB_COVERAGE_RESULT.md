# R236 Loose-Probability Coverage Result

## ARIS Status
- Stage: original-val full-val mask-level development
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: yes, isolated output only
- Decision: `R236 = best_safe_coverage_so_far; still_micro_corrections_not_claim_ready`

## Purpose
R235 improved application coverage to all `96` original-val anchors and edited
`15/96` images, but many candidates were still rejected for low seam-probability
scores. R236 tests a broader original-val-only candidate and probability
configuration while keeping the final safety gate strict:

- zero Recall loss;
- no component count MAE worsening;
- positive boundary/gap movement;
- no accepted over-erosion under development diagnostics.

Clean-test-v2 was not used.

## Outputs
- Summary: `outputs/analysis/r236_merged_seam_prob_allanchors_looseprob_hgb_summary.json`
- Per-image CSV: `outputs/analysis/r236_merged_seam_prob_allanchors_looseprob_hgb_per_image.csv`
- Candidate CSV: `outputs/analysis/r236_merged_seam_prob_allanchors_looseprob_hgb_candidates.csv`
- Mask dir: `outputs/ablations_variants/r236_merged_seam_prob_allanchors_looseprob_hgb/TSRS_RSNA-Epiphysis/val/masks`
- Log: `outputs/bridge_logs/r236_merged_seam_prob_allanchors_looseprob_hgb.log`
- Visual audit: `outputs/analysis/r236_merged_seam_prob_allanchors_looseprob_hgb_visual_audit/index.html`

## Key Settings
Relative to R235:

- `--ridge-prob-thresholds 0.60,0.70,0.80`
- `--ridge-action-fracs 0.15,0.20,0.30`
- `--action-fracs 0.20,0.25,0.35,0.50`
- `--seam-prob-threshold 0.30`
- `--seam-prob-p90-threshold 0.70`
- `--max-ridges-per-component 6`
- `--max-candidates-per-image 144`
- final `--recall-tol 0`

## Result
Mean over `96` original-val anchor images:

- edited images: `20`
- Dice delta: `+0.000046`
- IoU delta: `+0.000073`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000149`
- Boundary F1 delta: `+0.000198`
- gap-region FP rate delta: `-0.000176`
- component count MAE delta: `+0.000000`
- mean cut pixels: `4.062500`

Accepted family split:

- probability ridge: `12`
- corridor: `8`

All accepted cases have:

- `cut_gt_fg_frac = 0.0`
- `cut_gt_gap_frac = 1.0`
- `delta_recall = 0.0`

## Comparison
| Run | Evaluated | Edited | Dice Δ | IoU Δ | Recall Δ | Boundary IoU Δ | Boundary F1 Δ | Gap FP Δ | Component MAE Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R235 all-anchor HGB | 96 | 15 | +0.000037 | +0.000058 | +0.000000 | +0.000135 | +0.000182 | -0.000143 | +0.000000 |
| R236 loose-prob HGB | 96 | 20 | +0.000046 | +0.000073 | +0.000000 | +0.000149 | +0.000198 | -0.000176 | +0.000000 |

R236 adds seven accepted cases beyond R235:

- `10624.png`
- `1736.png`
- `3058.png`
- `3392.png`
- `3678.png`
- `6606.png`
- `8253.png`

## Visual Audit
Generated `20` panels:

- `outputs/analysis/r236_merged_seam_prob_allanchors_looseprob_hgb_visual_audit/index.html`

Inspected representative new cases:

- `10624.png`: corridor cut removes a small seam/background FP strip without
  visible漏分.
- `1736.png`: ridge cut lies in a narrow background seam; no visible foreground
  deletion.
- `3058.png`: small gap/FP correction; boundary metrics are unchanged, so this
  is safe but weak evidence.
- `3678.png`: corridor cut sits between adjacent structures and appears
  anatomically plausible.

Visual audit supports safety, but the edits are still mostly small FP/gap
corrections rather than clear component-merge repairs.

## Interpretation
R236 improves safe coverage without sacrificing overlap:

- edited coverage increases from `15/96` to `20/96`;
- Dice, IoU, Boundary IoU, Boundary F1, and gap FP all move in the intended
  direction;
- Recall remains unchanged;
- accepted cuts remain pure gap/background under development diagnostics.

However, R236 still does not fully satisfy the project target:

- component count MAE remains unchanged;
- the average gains are small;
- many accepted edits are local FP cleanup rather than strong bone-seam
  de-adhesion;
- no clean-test-v2 evaluation should be run yet.

## Decision
Keep R236 as the current best safe full original-val checkpoint. The next route
should specifically target component merge / connected-component errors instead
of only increasing tiny gap-FP edits. A useful next experiment is a candidate
family that explicitly rewards splitting predicted components only when Recall
and component-count MAE gates remain safe.
