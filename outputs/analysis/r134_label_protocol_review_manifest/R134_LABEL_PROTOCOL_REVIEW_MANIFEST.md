# R134 Label/Protocol Review Manifest

This is a non-GPU curation artifact. It does not modify the original dataset and does not use the new reannotated test split.

## Raw Data Table

| Group | Count | Notes |
| --- | ---: | --- |
| clean-test-v2 reference | 24 | diagnostic only; never train/tune on these labels |
| train review queue | 64 | candidates for isolated corrected/verified train variant after human review |
| val audit queue | 6 | protocol consistency slice, not final success metric |

## Key Findings

1. R127 already rejects simple resampling and broad filtering: `manual_label_protocol_review_supported=True`, `next_gpu_supported_without_new_labels=False`.
2. Train review queue priorities: `{'P0': 25, 'P1': 38, 'P2': 1}`. P0 contains bridge/boundary cases most aligned with R110/R132 failures.
3. Train prototype counts: `{'under_separated_bridge': 16, 'severe_boundary_shift': 5, 'mixed_boundary_bridge': 4, 'candidate_shared_failure': 9, 'underreach_high_precision': 18, 'overmask_low_precision': 7, 'all_hard': 4, 'candidate_fixable': 1}`. The queue is still not candidate-fixable dominated, so the next run should be a verified-label/protocol variant, not candidate fusion.

## Suggested Next Experiment

R134 should remain non-GPU until human decisions are filled. After review, create an isolated dataset variant only from train/val decisions; keep clean-test-v2 solely for final evaluation.

Gate for a later GPU run: at least 20 train cases reviewed, with either >=8 confirmed label corrections or >=8 protocol-clear hard examples in the P0/P1 categories.

## Top Train Review Items

| Rank | Image | Priority | Prototype | Score | Action | Panel |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | 3796.png | `P0` | `under_separated_bridge` | 0.358 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/02_train_3796_candidate.jpg) |
| 2 | 2904.png | `P0` | `under_separated_bridge` | 0.375 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/03_train_2904_candidate.jpg) |
| 3 | 2516.png | `P0` | `under_separated_bridge` | 0.380 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/04_train_2516_candidate.jpg) |
| 4 | 2393.png | `P0` | `under_separated_bridge` | 0.398 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/05_train_2393_candidate.jpg) |
| 5 | 1387.png | `P0` | `severe_boundary_shift` | 0.401 | check contour placement and label extent; mark protocol ambiguity if boundary is subjective | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/06_train_1387_candidate.jpg) |
| 6 | 14302.png | `P0` | `mixed_boundary_bridge` | 0.432 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/11_train_14302_candidate.jpg) |
| 7 | 3906.png | `P0` | `under_separated_bridge` | 0.442 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/13_train_3906_candidate.jpg) |
| 8 | 2976.png | `P0` | `under_separated_bridge` | 0.466 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/19_train_2976_candidate.jpg) |
| 9 | 4140.png | `P0` | `under_separated_bridge` | 0.474 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/21_train_4140_candidate.jpg) |
| 10 | 2741.png | `P0` | `under_separated_bridge` | 0.488 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/27_train_2741_candidate.jpg) |
| 11 | 2509.png | `P0` | `under_separated_bridge` | 0.494 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/29_train_2509_candidate.jpg) |
| 12 | 15339.png | `P0` | `mixed_boundary_bridge` | 0.513 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 13 | 2906.png | `P0` | `under_separated_bridge` | 0.518 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 14 | 3053.png | `P0` | `under_separated_bridge` | 0.520 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 15 | 3003.png | `P0` | `under_separated_bridge` | 0.522 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 16 | 1803.png | `P0` | `severe_boundary_shift` | 0.527 | check contour placement and label extent; mark protocol ambiguity if boundary is subjective |  |
| 17 | 3547.png | `P0` | `under_separated_bridge` | 0.534 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 18 | 10015.png | `P0` | `mixed_boundary_bridge` | 0.535 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 19 | 3499.png | `P0` | `severe_boundary_shift` | 0.543 | check contour placement and label extent; mark protocol ambiguity if boundary is subjective |  |
| 20 | 3495.png | `P0` | `under_separated_bridge` | 0.549 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 21 | 6488.png | `P0` | `mixed_boundary_bridge` | 0.553 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 22 | 1750.png | `P0` | `severe_boundary_shift` | 0.554 | check contour placement and label extent; mark protocol ambiguity if boundary is subjective |  |
| 23 | 3521.png | `P0` | `under_separated_bridge` | 0.564 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 24 | 3942.png | `P0` | `under_separated_bridge` | 0.569 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries |  |
| 25 | 1794.png | `P0` | `severe_boundary_shift` | 0.573 | check contour placement and label extent; mark protocol ambiguity if boundary is subjective |  |
| 26 | 2103.png | `P1` | `candidate_shared_failure` | 0.322 | review label/protocol or collect stronger examples; candidates do not provide a reliable correction | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/01_train_2103_candidate.jpg) |
| 27 | 15440.png | `P1` | `underreach_high_precision` | 0.413 | check whether GT includes narrow/small epiphysis extent that models consistently miss | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/08_train_15440_candidate.jpg) |
| 28 | 2108.png | `P1` | `overmask_low_precision` | 0.431 | check whether background/carpal/wrist tissue is included by mistake or whether label protocol allows broader extent | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/10_train_2108_candidate.jpg) |
| 29 | 14050.png | `P1` | `underreach_high_precision` | 0.441 | check whether GT includes narrow/small epiphysis extent that models consistently miss | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/12_train_14050_candidate.jpg) |
| 30 | 15362.png | `P1` | `underreach_high_precision` | 0.447 | check whether GT includes narrow/small epiphysis extent that models consistently miss | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/14_train_15362_candidate.jpg) |

## Val Protocol Audit Items

| Rank | Image | Priority | Prototype | Score | Action | Panel |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | 2024.png | `P0` | `under_separated_bridge` | 0.422 | check whether adjacent epiphysis/carpal regions should be separated; confirm bridge pixels and instance boundaries | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/09_val_2024_candidate.jpg) |
| 2 | 13454.png | `P1` | `underreach_high_precision` | 0.413 | check whether GT includes narrow/small epiphysis extent that models consistently miss | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/07_val_13454_candidate.jpg) |
| 3 | 2044.png | `P1` | `candidate_shared_failure` | 0.498 | review label/protocol or collect stronger examples; candidates do not provide a reliable correction | [panel](outputs/analysis/r125_hard_case_curation_manifest/panels/31_val_2044_candidate.jpg) |
| 4 | 7269.png | `P1` | `underreach_high_precision` | 0.515 | check whether GT includes narrow/small epiphysis extent that models consistently miss |  |
| 5 | 7253.png | `P1` | `underreach_high_precision` | 0.534 | check whether GT includes narrow/small epiphysis extent that models consistently miss |  |
| 6 | 7312.png | `P1` | `all_hard` | 0.552 | review val label/protocol consistency before using in training |  |

## Clean-Test Reference Cases

These cases explain the target failure modes but must not be used for training or threshold selection.

| Rank | Image | Tag | Dice | Boundary IoU | Panel |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | 4818.png | `candidate_shared_failure` | 0.853 | 0.194 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/01_4818_audit.jpg) |
| 2 | 12936.png | `candidate_fixable` | 0.859 | 0.308 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/02_12936_audit.jpg) |
| 3 | 3665.png | `under_separated_bridge` | 0.876 | 0.198 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/03_3665_audit.jpg) |
| 4 | 9194.png | `underreach_high_precision` | 0.878 | 0.276 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/04_9194_audit.jpg) |
| 5 | 2136.png | `mixed_boundary_bridge` | 0.880 | 0.235 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/05_2136_audit.jpg) |
| 6 | 8435.png | `underreach_high_precision` | 0.885 | 0.305 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/06_8435_audit.jpg) |
| 7 | 7484.png | `candidate_fixable` | 0.887 | 0.286 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/07_7484_audit.jpg) |
| 8 | 1897.png | `severe_boundary_shift` | 0.890 | 0.100 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/08_1897_audit.jpg) |
| 9 | 2050.png | `under_separated_bridge` | 0.890 | 0.174 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/09_2050_audit.jpg) |
| 10 | 2075.png | `under_separated_bridge` | 0.892 | 0.171 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/10_2075_audit.jpg) |
| 11 | 4099.png | `severe_boundary_shift` | 0.896 | 0.139 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/11_4099_audit.jpg) |
| 12 | 4078.png | `underreach_high_precision` | 0.897 | 0.124 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/12_4078_audit.jpg) |
| 13 | 4319.png | `overmask_low_precision` | 0.899 | 0.175 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/13_4319_audit.jpg) |
| 14 | 1468.png | `under_separated_bridge` | 0.900 | 0.174 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/14_1468_audit.jpg) |
| 15 | 3900.png | `severe_boundary_shift` | 0.900 | 0.137 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/15_3900_audit.jpg) |
| 16 | 2982.png | `candidate_fixable` | 0.902 | 0.215 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/16_2982_audit.jpg) |
| 17 | 2495.png | `candidate_shared_failure` | 0.902 | 0.195 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/17_2495_audit.jpg) |
| 18 | 1784.png | `candidate_fixable` | 0.903 | 0.251 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/18_1784_audit.jpg) |
| 19 | 1789.png | `severe_boundary_shift` | 0.905 | 0.123 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/19_1789_audit.jpg) |
| 20 | 7397.png | `mixed_boundary_bridge` | 0.905 | 0.260 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/20_7397_audit.jpg) |
| 21 | 1891.png | `overmask_low_precision` | 0.910 | 0.108 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/21_1891_audit.jpg) |
| 22 | 2168.png | `under_separated_bridge` | 0.915 | 0.251 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/22_2168_audit.jpg) |
| 23 | 3660.png | `under_separated_bridge` | 0.906 | 0.232 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/23_3660_audit.jpg) |
| 24 | 15578.png | `underreach_high_precision` | 0.906 | 0.242 | [panel](outputs/analysis/r122_r110_hard_case_audit_pack/panels/24_15578_audit.jpg) |
