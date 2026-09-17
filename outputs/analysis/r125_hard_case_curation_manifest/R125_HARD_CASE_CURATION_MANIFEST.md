# R125 Hard-Case Curation Manifest

This manifest is an audit/curation artifact. It does not modify the original dataset and does not use the new reannotated test split.

## Summary

- Source dataset: `TSRS_RSNA-Epiphysis`
- Hard train candidates: `64`
- Hard val audit candidates: `6`
- Train prototype counts: `{'under_separated_bridge': 16, 'underreach_high_precision': 18, 'candidate_shared_failure': 9, 'overmask_low_precision': 7, 'severe_boundary_shift': 5, 'mixed_boundary_bridge': 4, 'all_hard': 4, 'candidate_fixable': 1}`
- Val prototype counts: `{'under_separated_bridge': 1, 'underreach_high_precision': 3, 'candidate_shared_failure': 1, 'all_hard': 1}`

## Decision

Visual spot checks of R124 top panels confirm train-side analogues of low-contrast wrist/carpal multi-component cases and bridge/boundary-fragment modes.
The pool is not dominated by `candidate_fixable`, so the next GPU run should not be another candidate-fusion readout. A justified next model change would need to use this manifest for hard-case sampling or a label/layout protocol variant, with clean-test-v2 kept only for final evaluation.

## Recommended Use

- Review the listed panels and mark any label-protocol ambiguity before training.
- If training is launched, oversample `hard_train` in an isolated R126-style experiment instead of changing the original dataset.
- Keep `hard_val_audit` as a diagnostic slice, not as the final success metric.

## Top Hard Train

| Rank | Image | Prototype | Score | Reason | Panel |
| ---: | --- | --- | ---: | --- | --- |
| 1 | 3796.png | `under_separated_bridge` | 0.358 | r124_hardcase_similarity_train_candidate | [panel](panels/02_train_3796_candidate.jpg) |
| 2 | 2904.png | `under_separated_bridge` | 0.375 | r124_hardcase_similarity_train_candidate | [panel](panels/03_train_2904_candidate.jpg) |
| 3 | 2516.png | `under_separated_bridge` | 0.380 | r124_hardcase_similarity_train_candidate | [panel](panels/04_train_2516_candidate.jpg) |
| 4 | 2393.png | `under_separated_bridge` | 0.398 | r124_hardcase_similarity_train_candidate | [panel](panels/05_train_2393_candidate.jpg) |
| 5 | 3906.png | `under_separated_bridge` | 0.442 | r124_hardcase_similarity_train_candidate | [panel](panels/13_train_3906_candidate.jpg) |
| 6 | 2976.png | `under_separated_bridge` | 0.466 | r124_hardcase_similarity_train_candidate | [panel](panels/19_train_2976_candidate.jpg) |
| 7 | 4140.png | `under_separated_bridge` | 0.474 | r124_hardcase_similarity_train_candidate | [panel](panels/21_train_4140_candidate.jpg) |
| 8 | 2741.png | `under_separated_bridge` | 0.488 | r124_hardcase_similarity_train_candidate | [panel](panels/27_train_2741_candidate.jpg) |
| 9 | 2509.png | `under_separated_bridge` | 0.494 | r124_hardcase_similarity_train_candidate | [panel](panels/29_train_2509_candidate.jpg) |
| 10 | 2906.png | `under_separated_bridge` | 0.518 | r124_hardcase_similarity_train_candidate | [panel](panels/40_train_2906_candidate.jpg) |
| 11 | 3053.png | `under_separated_bridge` | 0.520 | r124_hardcase_similarity_train_candidate | [panel](panels/42_train_3053_candidate.jpg) |
| 12 | 3003.png | `under_separated_bridge` | 0.522 | r124_hardcase_similarity_train_candidate | [panel](panels/44_train_3003_candidate.jpg) |
| 13 | 3547.png | `under_separated_bridge` | 0.534 | r124_hardcase_similarity_train_candidate | [panel](panels/51_train_3547_candidate.jpg) |
| 14 | 3495.png | `under_separated_bridge` | 0.549 | r124_hardcase_similarity_train_candidate | [panel](panels/63_train_3495_candidate.jpg) |
| 15 | 3521.png | `under_separated_bridge` | 0.564 | r124_hardcase_similarity_train_candidate | [panel](panels/74_train_3521_candidate.jpg) |
| 16 | 3942.png | `under_separated_bridge` | 0.569 | r124_hardcase_similarity_train_candidate | [panel](panels/77_train_3942_candidate.jpg) |
| 17 | 15440.png | `underreach_high_precision` | 0.413 | r124_hardcase_similarity_train_candidate | [panel](panels/08_train_15440_candidate.jpg) |
| 18 | 14050.png | `underreach_high_precision` | 0.441 | r124_hardcase_similarity_train_candidate | [panel](panels/12_train_14050_candidate.jpg) |
| 19 | 15362.png | `underreach_high_precision` | 0.447 | r124_hardcase_similarity_train_candidate | [panel](panels/14_train_15362_candidate.jpg) |
| 20 | 4878.png | `underreach_high_precision` | 0.448 | r124_hardcase_similarity_train_candidate | [panel](panels/15_train_4878_candidate.jpg) |
| 21 | 15003.png | `underreach_high_precision` | 0.450 | r124_hardcase_similarity_train_candidate | [panel](panels/16_train_15003_candidate.jpg) |
| 22 | 8717.png | `underreach_high_precision` | 0.460 | r124_hardcase_similarity_train_candidate | [panel](panels/18_train_8717_candidate.jpg) |
| 23 | 15380.png | `underreach_high_precision` | 0.467 | r124_hardcase_similarity_train_candidate | [panel](panels/20_train_15380_candidate.jpg) |
| 24 | 15317.png | `underreach_high_precision` | 0.478 | r124_hardcase_similarity_train_candidate | [panel](panels/25_train_15317_candidate.jpg) |
| 25 | 5925.png | `underreach_high_precision` | 0.497 | r124_hardcase_similarity_train_candidate | [panel](panels/30_train_5925_candidate.jpg) |
| 26 | 15177.png | `underreach_high_precision` | 0.498 | r124_hardcase_similarity_train_candidate | [panel](panels/32_train_15177_candidate.jpg) |
| 27 | 11779.png | `underreach_high_precision` | 0.516 | r124_hardcase_similarity_train_candidate | [panel](panels/39_train_11779_candidate.jpg) |
| 28 | 10621.png | `underreach_high_precision` | 0.522 | r124_hardcase_similarity_train_candidate | [panel](panels/45_train_10621_candidate.jpg) |
| 29 | 10910.png | `underreach_high_precision` | 0.522 | r124_hardcase_similarity_train_candidate | [panel](panels/46_train_10910_candidate.jpg) |
| 30 | 9789.png | `underreach_high_precision` | 0.526 | r124_hardcase_similarity_train_candidate | [panel](panels/48_train_9789_candidate.jpg) |
| 31 | 5692.png | `underreach_high_precision` | 0.534 | r124_hardcase_similarity_train_candidate | [panel](panels/52_train_5692_candidate.jpg) |
| 32 | 15509.png | `underreach_high_precision` | 0.537 | r124_hardcase_similarity_train_candidate | [panel](panels/55_train_15509_candidate.jpg) |
| 33 | 15587.png | `underreach_high_precision` | 0.549 | r124_hardcase_similarity_train_candidate | [panel](panels/64_train_15587_candidate.jpg) |
| 34 | 15174.png | `underreach_high_precision` | 0.558 | r124_hardcase_similarity_train_candidate | [panel](panels/71_train_15174_candidate.jpg) |
| 35 | 2103.png | `candidate_shared_failure` | 0.322 | r124_hardcase_similarity_train_candidate | [panel](panels/01_train_2103_candidate.jpg) |
| 36 | 2163.png | `candidate_shared_failure` | 0.460 | r124_hardcase_similarity_train_candidate | [panel](panels/17_train_2163_candidate.jpg) |
| 37 | 3181.png | `candidate_shared_failure` | 0.475 | r124_hardcase_similarity_train_candidate | [panel](panels/22_train_3181_candidate.jpg) |
| 38 | 2123.png | `candidate_shared_failure` | 0.478 | r124_hardcase_similarity_train_candidate | [panel](panels/24_train_2123_candidate.jpg) |
| 39 | 2152.png | `candidate_shared_failure` | 0.493 | r124_hardcase_similarity_train_candidate | [panel](panels/28_train_2152_candidate.jpg) |
| 40 | 4339.png | `candidate_shared_failure` | 0.499 | r124_hardcase_similarity_train_candidate | [panel](panels/33_train_4339_candidate.jpg) |

## Hard Val Audit

| Rank | Image | Prototype | Score | Reason | Panel |
| ---: | --- | --- | ---: | --- | --- |
| 1 | 2024.png | `under_separated_bridge` | 0.422 | r124_hardcase_similarity_val_audit | [panel](panels/09_val_2024_candidate.jpg) |
| 2 | 13454.png | `underreach_high_precision` | 0.413 | r124_hardcase_similarity_val_audit | [panel](panels/07_val_13454_candidate.jpg) |
| 3 | 7269.png | `underreach_high_precision` | 0.515 | r124_hardcase_similarity_val_audit | [panel](panels/38_val_7269_candidate.jpg) |
| 4 | 7253.png | `underreach_high_precision` | 0.534 | r124_hardcase_similarity_val_audit | [panel](panels/53_val_7253_candidate.jpg) |
| 5 | 2044.png | `candidate_shared_failure` | 0.498 | r124_hardcase_similarity_val_audit | [panel](panels/31_val_2044_candidate.jpg) |
| 6 | 7312.png | `all_hard` | 0.552 | r124_hardcase_similarity_val_audit | [panel](panels/66_val_7312_candidate.jpg) |
