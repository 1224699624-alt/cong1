# R230 Strict Seam-Probability Safety Result

## ARIS Status
- Stage: original-val mask-level development
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: yes, isolated output only
- Decision: `R230 = cleanest_mask_level_positive_so_far; coverage still too low`

## Purpose
R229 produced the first mask-level positive result in the seam-probability
branch, but one accepted cut removed some GT foreground. R230 tests stricter
safety:

1. require zero Recall loss (`--recall-tol 0`);
2. disallow component count MAE worsening;
3. keep the seam-probability mean/p90 gates;
4. compare a strict baseline against a slightly expanded candidate generator.

This remains original-val development only. It is not clean-test-v2 or R201
final evidence.

## Implemented Through Existing Script
- Script: `scripts/run_r229_seam_prob_mask_editor.py`

R230 uses the same script with stricter/expanded parameters and isolated output
directories.

## Outputs

Strict recall baseline:
- Summary: `outputs/analysis/r230_seam_prob_strict_recall_pilot32_summary.json`
- Per-image CSV: `outputs/analysis/r230_seam_prob_strict_recall_pilot32_per_image.csv`
- Candidate CSV: `outputs/analysis/r230_seam_prob_strict_recall_pilot32_candidates.csv`
- Mask dir: `outputs/ablations_variants/r230_seam_prob_strict_recall_pilot32/TSRS_RSNA-Epiphysis/val/masks`

Strict recall expanded:
- Summary: `outputs/analysis/r230_seam_prob_strict_recall_expanded_pilot32_summary.json`
- Per-image CSV: `outputs/analysis/r230_seam_prob_strict_recall_expanded_pilot32_per_image.csv`
- Candidate CSV: `outputs/analysis/r230_seam_prob_strict_recall_expanded_pilot32_candidates.csv`
- Mask dir: `outputs/ablations_variants/r230_seam_prob_strict_recall_expanded_pilot32/TSRS_RSNA-Epiphysis/val/masks`

## Strict Recall Baseline
Command key settings:
- `--recall-tol 0`
- `--corridor-radii 1`
- `--action-fracs 0.35`

Mean over 12 evaluated original-val images:
- edited images: `1`
- Dice delta: `+0.000009`
- IoU delta: `+0.000014`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000004`
- Boundary F1 delta: `+0.000006`
- gap-region FP rate delta: `-0.000043`
- component count MAE delta: `+0.000000`

Accepted cut:
- `13867.png`
- cut pixels: `12`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`
- hard-risk: `0`
- over-erosion proxy: `0`

## Strict Recall Expanded
Command key settings:
- `--recall-tol 0`
- `--corridor-radii 1,2`
- `--action-fracs 0.25,0.35,0.50`

Mean over 12 evaluated original-val images:
- edited images: `2`
- Dice delta: `+0.000012`
- IoU delta: `+0.000020`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000023`
- Boundary F1 delta: `+0.000028`
- gap-region FP rate delta: `-0.000060`
- component count MAE delta: `+0.000000`
- written masks: `12`
- per-image rows: `12`

Accepted cuts:

### `13867.png`
- cut pixels: `8`
- corridor radius: `2`
- action fraction: `0.25`
- Dice delta: `+0.000073`
- IoU delta: `+0.000114`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000010`
- Boundary F1 delta: `+0.000015`
- gap-region FP delta: `-0.000341`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

### `15067.png`
- cut pixels: `10`
- corridor radius: `1`
- action fraction: `0.25`
- Dice delta: `+0.000073`
- IoU delta: `+0.000130`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000270`
- Boundary F1 delta: `+0.000325`
- gap-region FP delta: `-0.000377`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

## Comparison To R229
R229:
- edited images: `2`
- Recall delta: `-0.000005`
- Boundary IoU delta: `+0.000006`
- gap FP delta: `-0.000074`
- one accepted cut had GT foreground fraction `0.285714`

R230 strict expanded:
- edited images: `2`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000023`
- gap FP delta: `-0.000060`
- both accepted cuts have GT foreground fraction `0.000000`

R230 therefore improves safety and boundary gain over R229 while keeping the
same edited-image count in this pilot. Gap-FP gain is slightly smaller than
R229 but still positive.

## Interpretation
R230 is the cleanest mask-level result so far in this branch:
- no clean-test-v2 use;
- isolated original-val masks written;
- no Recall loss;
- no component count MAE worsening;
- no accepted over-erosion proxy;
- both accepted cuts are pure GT gap cuts in diagnostic analysis;
- Dice/IoU/Boundary/gap-FP all move in the intended direction.

The limitation is coverage:
- only `2/12` evaluated images were edited;
- this is a selected original-val pilot, not full-val and not R201;
- the method still depends on R226-style seed-pair candidates, which limits
coverage.

## Decision
Do not promote to clean-test-v2. Keep R230 strict expanded as the current best
safe development checkpoint for the seam-probability branch.

## Next Step
R231 should focus on coverage while preserving R230 safety:

1. Generate seam-probability ridge candidates directly from the probability map
   instead of relying only on seed-pair corridors.
2. Keep R230 safety gates:
   - zero Recall loss on original val;
   - no component count MAE worsening;
   - no over-erosion proxy;
   - positive boundary/gap-FP deltas.
3. Add hard-case visualization for `13867.png` and `15067.png` to confirm that
   the edits visibly reduce seam adhesion without漏分.
4. Only after full original-val mask-level evidence should any locked model be
   considered for R201 clean-test-v2 final evaluation.
