# R256 Scale-Invariant Pair Prior Full-Val Result

## Status

- Completed: 2026-07-14 19:13
- Scope: `TSRS_RSNA-Epiphysis_contrast_v1` original train/val only
- Runs: geometry-only and development-conditioned, seeds 256/257/258, 4 fixed epochs
- Validation pairs: 3,670 balanced pairs over all 96 cases
- clean-test-v2 used: no
- Articular-Surface used: no
- Official decision: `no_go_prior_not_validated`

## Three-seed fixed-epoch-4 means

| Metric | Geometry only | Development conditioned | Delta |
|---|---:|---:|---:|
| AUROC | 0.978492 | 0.983493 | +0.005001 |
| Average precision | 0.979860 | 0.984319 | +0.004459 |
| Close-pair recall | 0.740490 | 0.867420 | +0.126930 |
| Contact-pair recall | 0.664052 | 0.783007 | +0.118954 |
| Same-instance specificity | 0.898638 | 0.900999 | +0.002361 |
| Positive heatmap soft Dice | 0.213241 | 0.237115 | +0.023875 |

Development conditioning improved case-level close recall by `+0.109957` with paired bootstrap 95% CI `[+0.076437, +0.140982]`. Case-level AUROC delta was `+0.001778`, 95% CI `[-0.004374, +0.010749]`.

## Coverage and robustness

- All 885 relative-close and all 255 contact pairs were included: coverage `1.0/1.0`.
- Heat target coverage: `0.988556`; contact heat coverage: `0.976471`.
- Proposal geometry standardized mean difference: `0.344057` (gate <= 0.50).
- Under doubled proposal noise, development-conditioned AUROC `0.973946`, close recall `0.846328`, contact recall `0.762092`, specificity `0.887557`, heat Dice `0.213861`.
- Real metadata outperformed case-shuffled metadata in AUROC: `0.983493` vs `0.976923`.

## Why the official gate failed

Only `no_single_input_shortcut` failed. Removing the explicit geometry vector left AUROC nearly unchanged (`0.983348` vs full `0.983493`), below the preregistered required margin of `0.005`. Removing X-ray collapsed AUROC to `0.547292`; removing center maps reduced it to `0.795159`; metadata-only was chance (`0.499975`). Therefore the failed gate indicates redundancy of the explicit geometry vector with center maps, not that age/sex alone predicts the label. This interpretation is post-hoc and does not change the preregistered `no_go` decision.

## Decision

Do not integrate R256 directly into nnU-Net and do not use clean-test-v2. The mechanism result is promising but incomplete: age/sex conditioning substantially improves close/contact recall and heatmap quality while preserving specificity, but the proposed explicit geometry feature is redundant and fixed-threshold seed variability remains material. A next experiment must be preregistered separately and should simplify to X-ray + candidate-center maps + development conditioning, then replace GT-jitter proposals with an image-only center/scale detector.
