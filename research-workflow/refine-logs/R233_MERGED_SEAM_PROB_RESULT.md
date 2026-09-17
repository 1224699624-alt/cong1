# R233 Merged Seam-Probability Result

## ARIS Status
- Stage: original-val mask-level development
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: yes, isolated output only
- Decision: `R233 = best_local_pilot_so_far; needs_full_original_val_validation`

## Purpose
R230 showed safe corridor-based seam cuts but low coverage (`2/12`). R232
improved coverage with probability-ridge cuts (`3/12`). R233 merges both
candidate families under the same strict safety gate:

- zero Recall loss;
- no component count MAE worsening;
- positive Boundary IoU / Boundary F1 movement;
- lower gap-region FP rate;
- no accepted over-erosion proxy.

This tests whether the bottleneck is candidate coverage rather than the safety
gate itself.

## Implemented Artifacts
- Script: `scripts/run_r233_merged_seam_prob_mask_editor.py`
- Summary: `outputs/analysis/r233_merged_seam_prob_pilot32_summary.json`
- Per-image CSV: `outputs/analysis/r233_merged_seam_prob_pilot32_per_image.csv`
- Candidate CSV: `outputs/analysis/r233_merged_seam_prob_pilot32_candidates.csv`
- Mask dir: `outputs/ablations_variants/r233_merged_seam_prob_pilot32/TSRS_RSNA-Epiphysis/val/masks`
- Visual audit: `outputs/analysis/r233_merged_seam_prob_visual_audit/index.html`

## Key Settings
- `--limit-source 32`
- corridor candidates: `--corridor-radii 1,2`, `--action-fracs 0.25,0.35,0.50`
- ridge candidates: `--ridge-prob-thresholds 0.75,0.85`, `--ridge-action-fracs 0.20,0.30`, `--ridge-dilate-radii 0`
- `--max-components-per-image 1`
- `--max-ridges-per-component 4`
- `--max-candidates-per-image 96`
- `--recall-tol 0`

## R233 Pilot Result
Mean over the same `12` evaluated original-val pilot images:

- edited images: `4`
- Dice delta: `+0.000074`
- IoU delta: `+0.000130`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000380`
- Boundary F1 delta: `+0.000460`
- gap-region FP rate delta: `-0.000386`
- component count MAE delta: `-0.083333`
- mean cut pixels: `10.416667`
- written masks: `12`

Accepted cuts:

### `13867.png`
- family: `corridor`
- cut pixels: `12`
- Dice delta: `+0.000110`
- IoU delta: `+0.000171`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000046`
- Boundary F1 delta: `+0.000069`
- gap-region FP delta: `-0.000511`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

### `15040.png`
- family: `prob_ridge`
- cut pixels: `5`
- Dice delta: `+0.000035`
- IoU delta: `+0.000061`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000206`
- Boundary F1 delta: `+0.000250`
- gap-region FP delta: `-0.000189`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

### `15067.png`
- family: `prob_ridge`
- cut pixels: `87`
- Dice delta: `+0.000640`
- IoU delta: `+0.001134`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.003829`
- Boundary F1 delta: `+0.004594`
- gap-region FP delta: `-0.003278`
- component count MAE delta: `-1.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

### `1620.png`
- family: `prob_ridge`
- cut pixels: `21`
- Dice delta: `+0.000110`
- IoU delta: `+0.000193`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000482`
- Boundary F1 delta: `+0.000604`
- gap-region FP delta: `-0.000654`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

## Comparison
Same 12-image pilot:

| Run | Edited | Dice Δ | IoU Δ | Recall Δ | Boundary IoU Δ | Boundary F1 Δ | Gap FP Δ | Component MAE Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R230 strict corridor | 2/12 | +0.000012 | +0.000020 | +0.000000 | +0.000023 | +0.000028 | -0.000060 | +0.000000 |
| R232 ridge fast | 3/12 | +0.000036 | +0.000061 | +0.000000 | +0.000143 | +0.000175 | -0.000181 | +0.000000 |
| R233 merged | 4/12 | +0.000074 | +0.000130 | +0.000000 | +0.000380 | +0.000460 | -0.000386 | -0.083333 |

R233 is the strongest local pilot result so far: it improves coverage and all
targeted diagnostic directions while keeping Recall unchanged.

## Visual Audit
Generated panels:

- `outputs/analysis/r233_merged_seam_prob_visual_audit/panels/01_13867_r233_merged_seam_panel.jpg`
- `outputs/analysis/r233_merged_seam_prob_visual_audit/panels/02_15040_r233_merged_seam_panel.jpg`
- `outputs/analysis/r233_merged_seam_prob_visual_audit/panels/03_15067_r233_merged_seam_panel.jpg`
- `outputs/analysis/r233_merged_seam_prob_visual_audit/panels/04_1620_r233_merged_seam_panel.jpg`

Visual inspection:

- `13867.png`: small corridor cut in the seam/background channel, no visible
  foreground deletion.
- `15040.png`: very small ridge cut at a local false-positive seam/boundary
  region, no obvious漏分.
- `15067.png`: larger ridge cut, but still lies in the background gap between
  adjacent bones; visually consistent with reducing a false connection rather
  than eroding bone.
- `1620.png`: ridge cut lies in a background seam/FP region; no visible
  over-erosion in the crop.

## Limitations
- This remains a 12-image local pilot because local R110 original-val anchors
  are incomplete.
- `cut_gt_fg_frac` and `cut_gt_gap_frac` are development diagnostics, not
  deployable inputs.
- The large `15067.png` cut should be watched carefully in full-val validation
  because it dominates the mean component-count gain.
- No clean-test-v2 data was used. Do not promote to R201 final evaluation until
  full original-val behavior is stable.

## Decision
R233 should become the next development checkpoint, replacing R230/R232 as the
best local pilot. The next step is full original-val validation on the remote
workspace where all R110 val anchors are available. If full-val confirms the
same direction with no Recall loss and no over-erosion, then lock the method
and only then consider R201 clean-test-v2 final evaluation.
