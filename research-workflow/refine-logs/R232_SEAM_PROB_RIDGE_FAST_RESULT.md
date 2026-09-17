# R232 Seam-Probability Ridge Fast Result

## ARIS Status
- Stage: original-val mask-level development
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: yes, isolated output only
- Decision: `R232 = weak_positive_coverage_gain_over_R230; not_ready_for_clean_test_v2`

## Purpose
R230 was the safest mask-level seam-probability result so far, but coverage
was too low: only `2/12` evaluated pilot images were edited. R232 tests whether
candidate coverage can be improved by generating cuts directly from high
probability seam ridges instead of relying only on seed-pair corridors.

The safety goal is unchanged:

- zero Recall loss;
- no component count MAE worsening;
- positive Boundary IoU / Boundary F1 movement;
- lower gap-region FP rate;
- no accepted over-erosion proxy.

This is original-val development only. It is not R201 final evidence.

## Implemented Artifacts
- Script: `scripts/run_r232_seam_prob_ridge_mask_editor.py`
- Visual audit reused/genericized: `scripts/build_r231_r230_seam_hardcase_visualizations.py`

Main R232 fast pilot outputs:
- Summary: `outputs/analysis/r232_seam_prob_ridge_fast_pilot32_summary.json`
- Per-image CSV: `outputs/analysis/r232_seam_prob_ridge_fast_pilot32_per_image.csv`
- Candidate CSV: `outputs/analysis/r232_seam_prob_ridge_fast_pilot32_candidates.csv`
- Mask dir: `outputs/ablations_variants/r232_seam_prob_ridge_fast_pilot32/TSRS_RSNA-Epiphysis/val/masks`
- Visual audit: `outputs/analysis/r232_seam_prob_ridge_fast_visual_audit/index.html`

## Key Settings
- `--limit-source 32`
- `--max-components-per-image 1`
- `--ridge-prob-thresholds 0.75,0.85`
- `--ridge-action-fracs 0.20,0.30`
- `--ridge-dilate-radii 0`
- `--max-ridges-per-component 4`
- `--max-candidates-per-image 16`
- `--recall-tol 0`

The broader ridge grid was initially too slow locally and was stopped before a
complete JSON/CSV was written. The fast pilot is the comparable completed
result.

## R232 Fast Pilot Result
Mean over `12` evaluated original-val images:

- edited images: `3`
- Dice delta: `+0.000036`
- IoU delta: `+0.000061`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000143`
- Boundary F1 delta: `+0.000175`
- gap-region FP rate delta: `-0.000181`
- component count MAE delta: `+0.000000`
- mean cut pixels: `4.416667`
- written masks: `12`

Accepted cuts:

### `11852.png`
- cut pixels: `17`
- Dice delta: `+0.000139`
- IoU delta: `+0.000219`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000792`
- Boundary F1 delta: `+0.001004`
- gap-region FP delta: `-0.000738`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

### `13480.png`
- cut pixels: `15`
- Dice delta: `+0.000141`
- IoU delta: `+0.000243`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000210`
- Boundary F1 delta: `+0.000250`
- gap-region FP delta: `-0.000644`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

### `15067.png`
- cut pixels: `21`
- Dice delta: `+0.000154`
- IoU delta: `+0.000273`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000709`
- Boundary F1 delta: `+0.000852`
- gap-region FP delta: `-0.000791`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

## Comparison To R230 Strict Expanded
R230 strict expanded over the same `12` pilot images:

- edited images: `2`
- Dice delta: `+0.000012`
- IoU delta: `+0.000020`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000023`
- Boundary F1 delta: `+0.000028`
- gap-region FP rate delta: `-0.000060`
- component count MAE delta: `+0.000000`

R232 fast pilot:

- edited images: `3`
- Dice delta: `+0.000036`
- IoU delta: `+0.000061`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000143`
- Boundary F1 delta: `+0.000175`
- gap-region FP rate delta: `-0.000181`
- component count MAE delta: `+0.000000`

Interpretation: R232 improves pilot coverage from `2/12` to `3/12` and gives
larger mean boundary/gap gains without Recall or component-count degradation.
The effect is still small and pilot-limited.

## Visual Audit
Generated panels:

- `outputs/analysis/r232_seam_prob_ridge_fast_visual_audit/panels/01_11852_r232_ridge_fast_seam_panel.jpg`
- `outputs/analysis/r232_seam_prob_ridge_fast_visual_audit/panels/02_13480_r232_ridge_fast_seam_panel.jpg`
- `outputs/analysis/r232_seam_prob_ridge_fast_visual_audit/panels/03_15067_r232_ridge_fast_seam_panel.jpg`

Visual inspection:

- `11852.png`: cut is a small FP/background-region correction near the local
  seam area; no visible GT foreground deletion.
- `13480.png`: cut lies in the narrow background seam between adjacent
  structures and is visually consistent with reducing adhesion.
- `15067.png`: cut lies along the adjacent-bone seam/background channel and is
  visually consistent with the intended separation.

No panel shows obvious over-erosion or new漏分 in the cropped view.

## Limitations
- This is a `12` image local pilot, not full original-val.
- Local R110 val anchors are incomplete, so the pilot uses the same available
  subset as R230.
- The completed fast grid is intentionally conservative; broader grids are
  slower and need remote execution or more pruning.
- `cut_gt_fg_frac` and `cut_gt_gap_frac` are development diagnostics, not
  deployable inputs.
- No clean-test-v2 images were used and this must not be reported as R201 final
  evidence.

## Decision
R232 is a weak but real step forward over R230:

- coverage improved from `2/12` to `3/12`;
- all accepted cuts are pure GT-gap cuts under development diagnostics;
- Recall and component count MAE did not worsen;
- boundary and gap-FP metrics improved more than R230 on the same pilot;
- visual audit supports safety for the three edited cases.

Do not promote directly to clean-test-v2. The next step should be either:

1. run R232 fast/full-pruned ridge generation on full original-val where all
   anchors are available; or
2. merge R230 corridor candidates and R232 ridge candidates into a single
   conservative mask editor, still judged on original-val only.
