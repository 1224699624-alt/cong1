# R242 Candidate Crop Scorer Result

## ARIS Status
- Stage: original_val_candidate_crop_grouped_cv_diagnostic
- Dataset: `TSRS_RSNA-Epiphysis`
- Split: `val`
- Clean-test-v2 used: `False`
- Writes masks: `False`
- Decision: `no_go_candidate_crop_scorer_not_promoted`

## Data
- candidate rows: `8190`
- candidate images: `96`
- crops: `8190`
- safe useful labels: `765`
- risk labels: `7339`

## Best Low-Risk Threshold Row
- no non-empty threshold row

## Probability Audit
The default threshold grid selected no candidates because the risk head is
systematically high:

- `r242_useful_prob`: min `0.000868`, median `0.456185`, p95 `0.856532`, max `0.937368`
- `r242_risk_prob`: min `0.552814`, median `0.896219`, p95 `0.993089`, max `0.999959`

Safe-useful rows are not cleanly separated from risk rows:

- safe-useful rows: `765`
  - median useful prob `0.662506`
  - median risk prob `0.803339`
  - median cut GT foreground fraction `0.0`
  - median cut GT gap fraction `1.0`
- risk rows: `7339`
  - median useful prob `0.432457`
  - median risk prob `0.903785`
  - median cut GT foreground fraction `0.818182`
  - median cut GT gap fraction `0.142857`

This means the model learned some weak useful signal, but its risk head is not
calibrated or selective enough for deployment.

## Relaxed Threshold Audit
Relaxing thresholds does not rescue the route:

- best low-risk relaxed setting around `useful>=0.001`, `risk<=0.6` selects
  `15` rows/images, but only `1` is safe-useful and `14` are risk.
- mean deltas for that relaxed setting:
  - Dice/IoU not audited as a mask-level result
  - Recall `-0.000061890`
  - Boundary IoU `+0.000041781`
  - gap-region FP `-0.000079799`
  - mean cut GT foreground fraction `0.589180`
  - mean cut GT gap fraction `0.257582`
- a broader `risk<=0.7` frontier selects `51` rows/images, but only `12` are
  safe-useful and `39` are risk, with mean cut GT foreground fraction
  `0.508265`.

So the apparent gap/boundary gains are still mostly purchased by foreground
erosion risk. This violates the project gate: seam separation must not come
from over-erosion or missed bone.

## Promotion Gate
Passed checks:
- none

Failed checks:
- no non-empty threshold row
- relaxed thresholds select risk rows far more often than safe-useful rows
- relaxed thresholds have negative mean Recall
- relaxed thresholds cut too much GT foreground
- no evidence supports writing masks or using clean-test-v2

## Interpretation
R242 is a negative result for the current R239 candidate distribution plus
local crop CNN scorer.

The result is stronger than a simple "threshold too strict" failure. The full
candidate table contains many true safe-useful examples (`765`), but the scorer
does not separate them from the much larger foreground-cut risk population
(`7339`). Even when thresholds are relaxed below the original grid, selected
rows remain risk-heavy and foreground-heavy.

Decision: `R242=no_go_candidate_crop_scorer_not_promoted`.

Do not write masks, do not run clean-test-v2, and do not continue threshold
sweeps on the same R239 candidate distribution.

Next recommended route:

1. Return to the R237/R238 component-merge-targeted oracle structure, because
   it produced a stronger upper bound with clean gap cuts and component-count
   improvement.
2. Build a GT-free approximation of that cleaner candidate generator, rather
   than trying to classify all noisy R239 background-channel proposals.
3. Keep R236 as the best safe mask-level original-val checkpoint so far, and
   use R238-A as the oracle upper-bound target.
