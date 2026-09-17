# R229 Seam-Probability Mask Editor Result

## ARIS Status
- Stage: original-val mask-level development
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: yes, isolated output only
- Decision: `R229 = weak_mask_level_positive; expand only with stronger safety/coverage`

## Purpose
R228 showed that R224 oracle seam pixels can rank R226-style seam candidates.
R229 converts that signal into an actual original-val mask editor:

1. train the R228 seam-pixel scorer from R224 oracle pixels;
2. generate conservative R226-style background-channel cuts;
3. accept only high-probability cuts that satisfy fast metric guardrails;
4. write isolated edited masks for original `val`.

This is not clean-test-v2 evidence and not R201 final evidence.

## Implemented Artifact
- Script: `scripts/run_r229_seam_prob_mask_editor.py`

Pilot outputs:
- Summary: `outputs/analysis/r229_seam_prob_mask_editor_pilot32_summary.json`
- Per-image CSV: `outputs/analysis/r229_seam_prob_mask_editor_pilot32_per_image.csv`
- Candidate CSV: `outputs/analysis/r229_seam_prob_mask_editor_pilot32_candidates.csv`
- Mask dir: `outputs/ablations_variants/r229_seam_prob_mask_editor_pilot32/TSRS_RSNA-Epiphysis/val/masks`

Smoke outputs:
- `outputs/analysis/r229_seam_prob_mask_editor_smoke12_summary.json`
- `outputs/analysis/r229_seam_prob_mask_editor_smoke12_per_image.csv`
- `outputs/analysis/r229_seam_prob_mask_editor_smoke12_candidates.csv`

## Pilot32 Settings
- source order: `outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv`
- requested source limit: `32`
- actual evaluated images with local anchors: `12`
- max cuts per image: `1`
- seam probability mean threshold: `0.4`
- seam probability p90 threshold: `0.8`
- Dice tolerance: `5e-4`
- IoU tolerance: `8e-4`
- Recall tolerance: `1e-4`
- component MAE worsening: disallowed

Output integrity:
- per-image rows: `12`
- written masks: `12`
- edited images: `2`
- accepted candidates: `2`

## Pilot32 Mean Result
Mean over the 12 evaluated original-val images:

- Anchor Dice: `0.915344`
- R229 Dice: `0.915356`
- Dice delta: `+0.000013`
- IoU delta: `+0.000020`
- Recall delta: `-0.000005`
- Boundary IoU delta: `+0.000006`
- Boundary F1 delta: `+0.000009`
- gap-region FP rate delta: `-0.000074`
- component count MAE delta: `+0.000000`
- mean cut pixels: `2.166667`

## Edited Cases

### `13867.png`
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
- hard-risk: `0`
- over-erosion proxy: `0`

This is the cleanest positive example: the edit cuts only GT gap pixels and
improves overlap, boundary, and gap-FP diagnostics simultaneously.

### `15067.png`
- cut pixels: `14`
- Dice delta: `+0.000040`
- IoU delta: `+0.000071`
- Recall delta: `-0.000064`
- Boundary IoU delta: `+0.000032`
- Boundary F1 delta: `+0.000038`
- gap-region FP delta: `-0.000377`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.285714`
- cut GT gap fraction: `0.714286`
- hard-risk: `0`
- over-erosion proxy: `0`

This edit passes the current guardrails but is not a pure gap edit. It should be
reviewed visually before being trusted as a desirable anatomy correction.

## Interpretation
R229 is the first mask-level positive result in this seam-probability branch:
edited masks were written, and average Dice/IoU/boundary/gap diagnostics move in
the desired direction without component count MAE worsening.

However, the effect is still very small and coverage remains low:
- only `2/12` evaluated images were edited;
- only one accepted cut is a perfect GT-gap cut;
- one accepted cut removes some GT foreground, though below the over-erosion
proxy threshold and within recall tolerance;
- no full-val or clean-test-v2 conclusion is justified.

## Decision
Do not promote R229 to R201 or clean-test-v2. Treat it as a weak original-val
mask-level positive and continue improving safety/coverage.

## Next Step
R230 should either:

1. Run a stricter safety variant:
   - require predicted/diagnostic gap-purity proxy stronger than R229;
   - lower recall tolerance or require zero recall loss;
   - add a cut-shape/background-channel guard;
   - target only pure-gap-like candidates first.

2. Or improve coverage:
   - generate more seam-probability ridge candidates directly from probability
     maps, not only R226 seed-pair cuts;
   - still keep max cut fraction and anti-erosion guards.

Promotion gate for the next mask-level run:
- more edited images than R229 on original val;
- no accepted over-erosion proxy;
- ideally all accepted cuts have low foreground-removal proxy;
- Dice/IoU/Recall non-degradation;
- Boundary IoU/F1 improvement;
- gap-region FP reduction;
- component count MAE non-worsening;
- hard-case visualization for edited cases before any final-test use.
