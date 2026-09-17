# R241 Candidate-Feature Scorer Result

## ARIS Status
- Stage: original-val candidate-level scorer audit
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `no_go_feature_scorer_not_selective_enough`

## Purpose
R240 showed that pure GT-free geometry can open background channels and move
boundary/gap diagnostics, but it often cuts true bone foreground. R241-F0 tests
whether a lightweight scorer using the same deployable candidate-level features
can learn to separate:

- true seam/background cuts, and
- risky shallow foreground/bone cuts.

This is deliberately candidate-level only. It does not write masks and it does
not touch clean-test-v2.

## Implementation
Added:

- `scripts/train_r241_candidate_feature_scorer.py`

Input candidate table:

- `outputs/analysis/r239_gtfree_bg_channel_light_fullval_candidates.csv`

Features are GT-free numeric columns:

- background-channel geometry
- cut size/shape
- distance-to-anchor-interior features
- image intensity/gradient summaries
- seam-probability summaries

Labels use original-val GT only for audit/supervision:

- positive safe seam/background cut:
  - Dice >= `-5e-4`
  - IoU >= `-8e-4`
  - Recall >= `0`
  - Boundary IoU/F1 non-worse
  - gap FP improves
  - component count MAE non-worse
  - `cut_gt_fg_frac <= 0.25`
  - `cut_gt_gap_frac >= 0.75`
- risk:
  - high GT foreground cut fraction, recall loss, boundary degradation, or
    overlap degradation.

Grouped CV is split by image.

## Data
Candidate rows:

- rows: `280`
- images: `77`
- safe positives: `38`
- risk rows: `237`
- safe rate: `0.135714`
- risk rate: `0.846429`

The class imbalance is real and important: the candidate generator produces far
more risky foreground cuts than safe seam/background cuts.

## Results
HGB candidate-feature scorer:

- CV AUC: `0.467921`
- CV average precision: `0.121811`
- decision: `no_go_feature_scorer_not_selective_enough`

Best non-empty HGB threshold row:

- threshold: `0.8`
- selected rows/images: `3/3`
- safe: `0`
- risk: `3`
- Dice delta: `-0.000127`
- IoU delta: `-0.000195`
- Recall delta: `-0.000218`
- Boundary IoU delta: `-0.000327`
- Boundary F1 delta: `-0.000444`
- gap FP delta: `-0.000050`
- mean `cut_gt_fg_frac`: `0.833333`
- mean `cut_gt_gap_frac`: `0.166667`

Logistic-regression control:

- CV AUC: `0.535341`
- CV average precision: `0.142179`
- decision: `no_go_feature_scorer_not_selective_enough`

The logistic-control thresholds also fail. For example, at threshold `0.4` it
selects `52` rows with `10` safe and `40` risk, Recall decreases, and mean
`cut_gt_fg_frac` remains `0.527276`.

## Interpretation
R241-F0 is a strong negative result for tabular candidate-feature selection.

The scorer does not learn a reliable boundary between true seam/background cuts
and risky foreground cuts. In fact, the highest HGB probabilities are assigned
to several risky cuts. This means the currently available scalar features do
not encode enough local anatomical/image context.

This does **not** kill the reversed-topology route. R237/R238 still show a real
oracle upper bound. It does mean:

- more hand-tuned gates are low value;
- candidate-level scalar features are not enough;
- the next model must inspect local image/cut context directly or change the
  candidate generator so that its proposal distribution is cleaner.

## Next Step
Recommended R242 direction:

1. Build a local crop scorer over candidate actions.
2. Inputs should include:
   - grayscale image crop;
   - anchor mask crop;
   - candidate cut mask;
   - anchor interior distance transform;
   - local background distance/channel map;
   - optional seam-probability map.
3. Train with grouped image CV.
4. Positives:
   - R239/R240 safe pure-gap candidates;
   - R237/R238 oracle-safe cuts if they can be reconstructed as crops.
5. Hard negatives:
   - R240 selected risky foreground cuts;
   - high-probability R241 false positives.
6. Promotion gate:
   - selected risk must be far below selected safe;
   - Recall delta must stay `0`;
   - Boundary IoU/F1 and gap FP must beat R236 on original val;
   - visual audit must show seam separation without over-erosion or漏分.

Do not use clean-test-v2 for R242 model selection.
