# R316 / R318 Status

Date: 2026-07-27

## R316 safe relation selector

- Server: `connect.westc.seetacloud.com:43758`
- Dataset: full TSRS_RSNA-Epiphysis train/original-val, 875/96 images
- Test and clean-test-v2 used: no
- Train pairs: 14,491 positive + 14,491 negative
- Validation pairs: 1,778 positive + 1,778 negative
- Epochs: 8
- Final seam Soft Dice: 0.266040474
- Hard-negative count: 0

Final threshold examples:

| Threshold | Positive recall | Close-pair recall | Negative specificity |
|---:|---:|---:|---:|
| 0.50 | 0.959505 | 0.991247 | 0.836333 |
| 0.55 | 0.942632 | 0.986871 | 0.874578 |
| 0.60 | 0.913386 | 0.981400 | 0.903825 |
| 0.65 | 0.868954 | 0.967177 | 0.926884 |

Decision: no-go as evidence for a hard-negative-safe selector. Ordinary pair discrimination is usable, but the intended hard-negative subset is empty, so its specificity is undefined rather than proven. The seam heatmap is also weaker than the historical R258B/R260 value (~0.3393).

Local preserved bundle:

`outputs/artifact_bundles/r316_safe_selector_remote_43758_20260727`

## R318 full R260 prior retraining

- Screen: `r318_full_r260_prior`
- Dataset: full 875/96 train/original-val
- Model/config: exact R260/R258B fixed protocol
- Clean selection manifest: not used
- Test and clean-test-v2: locked
- Expected outputs:
  - `outputs/analysis/r318_full_r260_prior.json`
  - `outputs/analysis/r318_full_r260_prior_pairs.csv`
  - `outputs/pair_prior/r318_full_r260_prior`
  - `outputs/priors/r318_full_r260_relation`

## R316B full-data quantile hard-negative rerun

- Dataset: full 875 train / 96 original-val
- Hard-negative rule: closest 20% of negative proposal pairs
- Hard negatives: 2,899 train / 356 validation
- Final seam Soft Dice: 0.252114713
- Final threshold 0.50:
  - positive recall: 0.904386952
  - close-pair recall: 0.979212254
  - negative specificity: 0.907761530
  - hard-negative specificity: 0.581460674
- Final threshold 0.60:
  - positive recall: 0.804836895
  - close-pair recall: 0.943107221
  - negative specificity: 0.951631046
  - hard-negative specificity: 0.769662921

Decision: no-go. Quantile sampling fixed the empty hard-negative subset, but no epoch/threshold jointly achieved positive recall >= 0.90, close recall >= 0.97, ordinary specificity >= 0.90, and hard-negative specificity >= 0.75. The result exposes a genuine recall-safety conflict rather than a missing-supervision bug.

Local bundle: `outputs/artifact_bundles/r316b_full_quantile_remote_43758_20260727`
