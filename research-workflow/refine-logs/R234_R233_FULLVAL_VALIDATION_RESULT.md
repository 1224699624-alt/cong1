# R234 R233 Full-Val Validation Result

## ARIS Status
- Stage: original-val full-val mask-level validation
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: yes, isolated output only
- Decision: `R233_fullval_hgb = safe_weak_positive; coverage_still_insufficient_for_R201_promotion`

## Purpose
R233 was the best local 12-image pilot so far, merging R230 corridor candidates
and R232 probability-ridge candidates under the same strict safety gate. R234
validates R233 on the remote original-val workspace, where the R110 val anchor
set is more complete than local.

This remains development validation. It is not clean-test-v2 or R201 final
evidence.

## Remote Setup
- Remote workspace: `/home/shenzeyu/workspace/YOLO_SAM_generic_src`
- Environment: `conda activate yolo-sam-gpu`
- Remote val anchors available: `96`
- R224 candidate CSV available: yes
- Initial issue: `scikit-learn` was missing in `yolo-sam-gpu`.

Two runs were made:

1. `r233_merged_seam_prob_fullval`: numpy prototype fallback scorer, used only
   to confirm remote runnable behavior.
2. `r233_merged_seam_prob_fullval_hgb`: restored HGB scorer after installing
   `scikit-learn` in `yolo-sam-gpu`; this is the fair comparison to local R233.

The fallback support was added to:

- `scripts/audit_r228_oracle_pixel_seam_localizer.py`
- `scripts/audit_r226_bg_connectivity_seam_scorer.py`

R226/R228 now import without scikit-learn; HGB is still used automatically when
available.

## Fallback Run
Outputs:

- Summary: `outputs/analysis/r233_merged_seam_prob_fullval_summary.json`
- Per-image CSV: `outputs/analysis/r233_merged_seam_prob_fullval_per_image.csv`
- Candidate CSV: `outputs/analysis/r233_merged_seam_prob_fullval_candidates.csv`
- Mask dir: `outputs/ablations_variants/r233_merged_seam_prob_fullval/TSRS_RSNA-Epiphysis/val/masks`
- Log: `outputs/bridge_logs/r233_merged_seam_prob_fullval.log`

Result over `70` evaluated images:

- edited images: `1`
- Dice delta: `+0.000001`
- IoU delta: `+0.000002`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000019`
- Boundary F1 delta: `+0.000029`
- gap-region FP rate delta: `-0.000007`
- component count MAE delta: `+0.000000`

Interpretation: fallback scorer is safe but far too conservative / poorly
calibrated for R233. It is not the fair R233 result.

## HGB Full-Val Run
Outputs:

- Summary: `outputs/analysis/r233_merged_seam_prob_fullval_hgb_summary.json`
- Per-image CSV: `outputs/analysis/r233_merged_seam_prob_fullval_hgb_per_image.csv`
- Candidate CSV: `outputs/analysis/r233_merged_seam_prob_fullval_hgb_candidates.csv`
- Mask dir: `outputs/ablations_variants/r233_merged_seam_prob_fullval_hgb/TSRS_RSNA-Epiphysis/val/masks`
- Log: `outputs/bridge_logs/r233_merged_seam_prob_fullval_hgb.log`
- Visual audit: `outputs/analysis/r233_merged_seam_prob_fullval_hgb_visual_audit/index.html`

Result over `70` evaluated images:

- edited images: `9`
- Dice delta: `+0.000024`
- IoU delta: `+0.000041`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000089`
- Boundary F1 delta: `+0.000116`
- gap-region FP rate delta: `-0.000116`
- component count MAE delta: `+0.000000`
- mean cut pixels: `3.400000`
- candidate rows: `714`
- accepted candidates: `9`

Accepted candidates:

| Image | Family | Cut px | Dice Δ | BIoU Δ | BF1 Δ | Gap FP Δ | Component MAE Δ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `11852.png` | ridge | 31 | +0.000253 | +0.000740 | +0.000938 | -0.001086 | +0.000000 |
| `12827.png` | corridor | 5 | +0.000031 | +0.000191 | +0.000251 | -0.000183 | +0.000000 |
| `13436.png` | ridge | 86 | +0.000630 | +0.000959 | +0.001283 | -0.002550 | +0.000000 |
| `13867.png` | corridor | 8 | +0.000073 | +0.000010 | +0.000015 | -0.000341 | +0.000000 |
| `14041.png` | ridge | 41 | +0.000283 | +0.001176 | +0.001403 | -0.001611 | +0.000000 |
| `15446.png` | corridor | 20 | +0.000222 | +0.001815 | +0.002443 | -0.000940 | +0.000000 |
| `1620.png` | corridor | 19 | +0.000099 | +0.000744 | +0.000932 | -0.000592 | +0.000000 |
| `3249.png` | ridge | 24 | +0.000083 | +0.000416 | +0.000627 | -0.000673 | +0.000000 |
| `3602.png` | corridor | 4 | +0.000037 | +0.000178 | +0.000212 | -0.000179 | +0.000000 |

All accepted candidates have development-diagnostic:

- `cut_gt_fg_frac = 0.0`
- `cut_gt_gap_frac = 1.0`
- `delta_recall = 0.0`

## Visual Audit
Generated `9` panels at:

- `outputs/analysis/r233_merged_seam_prob_fullval_hgb_visual_audit/index.html`

Inspected representative panels:

- `13436.png`: largest cut (`86` px) lies in a narrow inter-bone background
  seam; no visible foreground deletion or obvious漏分.
- `15446.png`: corridor cut visually follows the adjacent-bone seam and
  improves boundary/gap diagnostics without visible over-erosion.
- `11852.png`: ridge cut removes FP/gap spillover in the seam area; no visible
  GT foreground deletion.
- `14041.png`: ridge cut lies in the seam/background channel and appears
  anatomically plausible.

The visual evidence supports safety for the accepted cases, but coverage is
still too low for a strong model claim.

## Comparison To Local Pilot
Local R233 pilot:

- `4/12` edited
- Boundary IoU delta `+0.000380`
- gap FP delta `-0.000386`
- component count MAE delta `-0.083333`

Remote full-val HGB:

- `9/70` edited
- Boundary IoU delta `+0.000089`
- gap FP delta `-0.000116`
- component count MAE delta `+0.000000`

The direction generalizes safely but weakly. The local component-count gain was
not confirmed on full-val.

## Limitations
- Evaluated `70` images because `source_names` is derived from the R224
  candidate CSV; remote has `96` val anchors. The remaining anchors are not
  source-listed by the current candidate pipeline.
- The accepted-edit coverage is only `9/70`.
- Full-val gains are positive but small.
- Component count MAE did not improve on full-val.
- This is still original-val development, not clean-test-v2 R201 evidence.

## Decision
R233 full-val HGB is a safe weak positive:

- It preserves Dice/IoU and Recall.
- It improves Boundary IoU, Boundary F1, and gap-region FP rate.
- Accepted cuts are pure gap/background under development diagnostics.
- Visual audit does not show obvious over-erosion or漏分.

However, it is not strong enough for R201 promotion because coverage and effect
size are too small, and component-count improvement did not generalize.

## Next Step
Do not run clean-test-v2 yet. The next development step should target coverage:

1. modify `source_names` / candidate generation so all `96` original-val anchor
   masks are evaluated, not only images appearing in R224 candidate CSV;
2. add a recall-safe lower probability threshold sweep on original-val only,
   because rejection analysis still shows many `low_mean_prob` and `recall_drop`
   candidates;
3. keep the strict final acceptance gate unchanged: zero Recall loss, no
   component MAE worsening, positive boundary/gap deltas.
