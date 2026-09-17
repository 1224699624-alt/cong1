# R228 Oracle Pixel Seam Localizer Result

## ARIS Status
- Stage: original-val candidate-level diagnostic
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `R228 = weak_go_for_real_localizer; no mask-level promotion yet`

## Purpose
R227 showed that R224 oracle supervision can help select safe R226-style
background-connectivity candidates, but coverage was only one clean accepted
candidate. R228 tests a more local signal: train a pixel seam scorer from R224
oracle seam pixels, then use its seam probability to rank R226-style GT-free
candidate cuts.

This is not a final model and not R201 evidence. It is a diagnostic test of
whether R224 oracle seam pixels contain a learnable local appearance/geometry
signal.

## Implemented Artifact
- Script: `scripts/audit_r228_oracle_pixel_seam_localizer.py`

The script:
- reconstructs R224 oracle seam pixels on original `val`;
- samples positive pixels from high-purity useful R224 seams;
- samples negative pixels from shallow non-seam anchor regions;
- trains a lightweight pixel classifier using image, anchor, distance,
  boundary, local mean, and coordinate features;
- scores R226-style background-connectivity candidates;
- reports candidate-level metrics only.

It does not:
- use clean-test-v2;
- write edited masks;
- use `delta_*` or candidate metrics as model features;
- claim mask-level or R201 improvement.

## Outputs
Smoke:
- `outputs/analysis/r228_oracle_pixel_seam_localizer_smoke12_summary.json`
- `outputs/analysis/r228_oracle_pixel_seam_localizer_smoke12_candidates.csv`
- `outputs/analysis/r228_oracle_pixel_seam_localizer_smoke12_grid.csv`

Pilot32:
- `outputs/analysis/r228_oracle_pixel_seam_localizer_pilot32_summary.json`
- `outputs/analysis/r228_oracle_pixel_seam_localizer_pilot32_candidates.csv`
- `outputs/analysis/r228_oracle_pixel_seam_localizer_pilot32_grid.csv`

## Pilot32 Result
Training sample summary:
- training images with R224 oracle pixels: `9`
- positive oracle pixels: `1000`
- sampled positives: `3184`
- sampled negatives: `6032`
- total samples: `9216`
- positive sample rate: `0.345486`

Candidate pool:
- candidate rows: `14`
- candidate images: `9`
- quick useful: `6/14`
- hard-risk: `8/14`
- safe-gap positive: `2/14`
- over-erosion proxy: `8/14`

Overall candidate pool remains noisy:
- Dice delta: `-0.000048`
- IoU delta: `-0.000082`
- Recall delta: `-0.000175`
- Boundary IoU delta: `-0.000197`
- Boundary F1 delta: `-0.000244`
- gap-region FP delta: `-0.000207`
- mean cut GT foreground fraction: `0.633716`

Best seam-probability gate:
- threshold: `0.4`
- accepted: `2`
- accepted images: `2`
- quick useful: `2/2`
- hard-risk: `0/2`
- over-erosion proxy: `0/2`
- safe-gap positive: `1/2`
- Dice delta: `+0.000075`
- IoU delta: `+0.000121`
- Recall delta: `-0.000032`
- Boundary IoU delta: `+0.000039`
- Boundary F1 delta: `+0.000053`
- gap-region FP delta: `-0.000444`
- component count MAE delta: `+0.000000`
- mean cut GT foreground fraction: `0.142857`
- mean cut GT gap fraction: `0.857143`

## Comparison To R227
R227 calibrated safe-gate:
- accepted: `1`
- hard-risk: `0`
- over-erosion proxy: `0`
- Boundary IoU delta: `+0.000046`
- gap FP delta: `-0.000511`

R228 pixel seam gate:
- accepted: `2`
- hard-risk: `0`
- over-erosion proxy: `0`
- Boundary IoU delta: `+0.000039`
- gap FP delta: `-0.000444`

R228 slightly improves clean candidate coverage while preserving the same safe
direction. However, the gain is still tiny and candidate-level only.

## Interpretation
R228 supports the idea that R224 oracle seam pixels contain a learnable local
signal. The seam probability ranks useful non-overerosive candidates above many
dangerous candidates, and the accepted set improves Dice/IoU, boundary metrics,
and gap FP simultaneously.

The bottleneck is still coverage and candidate generation:
- only `14` candidates were generated across `32` source images;
- only `2` candidates pass a safe seam-probability gate;
- no edited masks were written;
- no R201 final evaluation is justified.

## Decision
Do not promote R228 to mask-level R201 evaluation. Keep it as evidence that a
real localizer is worth building.

## Next Step
Proceed to R229 as a true localizer/mask-edit experiment:

1. Generate a seam probability map per R110 val image, not just score existing
   R226 candidates.
2. Use probability ridges to propose thin cuts directly.
3. Add explicit anti-erosion guards:
   - max cut fraction;
   - minimum background-channel support;
   - reject candidates with high predicted bone-interior score;
   - preserve Dice/IoU/Recall tolerances on original val.
4. Write isolated original-val masks only after candidate-level safeguards pass.
5. Evaluate mask-level original-val metrics before any clean-test-v2 use.

Promotion gate for R229:
- nonzero mask-level edited images on original val;
- Dice/IoU/Recall non-degradation;
- Boundary IoU/F1 improvement;
- gap-region FP reduction;
- component count MAE non-worsening;
- hard-case visualization showing reduced bone seam adhesion without
  over-erosion or漏分.
