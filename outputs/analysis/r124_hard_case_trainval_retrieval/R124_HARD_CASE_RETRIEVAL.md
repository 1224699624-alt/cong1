# R124 Hard-Case Train/Val Retrieval

This is a non-GPU data audit. It ranks original train/val labels by similarity to the R122/R123 hard-case audit set.

## Summary

- Target hard cases with local image/label features: `24`
- Prototypes: `all_hard(24), candidate_shared_failure(6), candidate_fixable(7), under_separated_bridge(6), underreach_high_precision(4), mixed_boundary_bridge(2), severe_boundary_shift(6), overmask_low_precision(7)`
- Top `80` candidate split counts: `{'train': 74, 'val': 6}`
- Top `80` nearest prototype counts: `{'candidate_shared_failure': 10, 'under_separated_bridge': 17, 'severe_boundary_shift': 5, 'underreach_high_precision': 25, 'overmask_low_precision': 7, 'mixed_boundary_bridge': 4, 'all_hard': 10, 'candidate_fixable': 2}`

## Interpretation

- Use these candidates for manual visual review and isolated data-variant design.
- Do not treat this as a new model result; it does not use clean-test-v2 labels for training or threshold selection.
- If the top candidates visibly match bridge/low-contrast hard cases, the next justified step is a hard-case curation manifest rather than another blind architecture run.

## Top Candidates

| Rank | Split | Image | Score | Prototype | Key Feature Deltas | Panel |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | train | 2103.png | 0.322 | `candidate_shared_failure` | aspect_median, layout_width, layout_cy_median | [panel](panels/01_train_2103_candidate.jpg) |
| 2 | train | 3796.png | 0.358 | `under_separated_bridge` | component_count, component_area_min, compactness_median | [panel](panels/02_train_3796_candidate.jpg) |
| 3 | train | 2904.png | 0.375 | `under_separated_bridge` | nearest_center_gap, compactness_median, thin_component_frac | [panel](panels/03_train_2904_candidate.jpg) |
| 4 | train | 2516.png | 0.380 | `under_separated_bridge` | aspect_median, layout_width, component_area_max | [panel](panels/04_train_2516_candidate.jpg) |
| 5 | train | 2393.png | 0.398 | `under_separated_bridge` | compactness_median, component_area_iqr, layout_width | [panel](panels/05_train_2393_candidate.jpg) |
| 6 | train | 1387.png | 0.401 | `severe_boundary_shift` | image_mean, bg_mean, layout_cy_median | [panel](panels/06_train_1387_candidate.jpg) |
| 7 | val | 13454.png | 0.413 | `underreach_high_precision` | layout_width, compactness_median, thin_component_frac | [panel](panels/07_val_13454_candidate.jpg) |
| 8 | train | 15440.png | 0.413 | `underreach_high_precision` | nearest_center_gap, compactness_median, image_mean | [panel](panels/08_train_15440_candidate.jpg) |
| 9 | val | 2024.png | 0.422 | `under_separated_bridge` | component_count, compactness_median, thin_component_frac | [panel](panels/09_val_2024_candidate.jpg) |
| 10 | train | 2108.png | 0.431 | `overmask_low_precision` | component_count, aspect_median, nearest_center_gap | [panel](panels/10_train_2108_candidate.jpg) |
| 11 | train | 14302.png | 0.432 | `mixed_boundary_bridge` | nearest_center_gap, layout_height, layout_cy_span | [panel](panels/11_train_14302_candidate.jpg) |
| 12 | train | 14050.png | 0.441 | `underreach_high_precision` | layout_width, nearest_center_gap, component_area_min | [panel](panels/12_train_14050_candidate.jpg) |
| 13 | train | 3906.png | 0.442 | `under_separated_bridge` | layout_width, component_count, compactness_median | [panel](panels/13_train_3906_candidate.jpg) |
| 14 | train | 15362.png | 0.447 | `underreach_high_precision` | nearest_center_gap, layout_width, bg_mean | [panel](panels/14_train_15362_candidate.jpg) |
| 15 | train | 4878.png | 0.448 | `underreach_high_precision` | perimeter_area_median, layout_cy_median, nearest_center_gap | [panel](panels/15_train_4878_candidate.jpg) |
| 16 | train | 15003.png | 0.450 | `underreach_high_precision` | fg_mean, component_area_min, image_mean | [panel](panels/16_train_15003_candidate.jpg) |
| 17 | train | 2163.png | 0.460 | `candidate_shared_failure` | aspect_median, layout_width, component_area_cv | [panel](panels/17_train_2163_candidate.jpg) |
| 18 | train | 8717.png | 0.460 | `underreach_high_precision` | compactness_median, component_area_cv, layout_cy_median | [panel](panels/18_train_8717_candidate.jpg) |
| 19 | train | 2976.png | 0.466 | `under_separated_bridge` | nearest_center_gap, compactness_median, component_area_iqr | [panel](panels/19_train_2976_candidate.jpg) |
| 20 | train | 15380.png | 0.467 | `underreach_high_precision` | layout_width, component_area_min, layout_height | [panel](panels/20_train_15380_candidate.jpg) |
| 21 | train | 4140.png | 0.474 | `under_separated_bridge` | compactness_median, aspect_median, layout_cy_median | [panel](panels/21_train_4140_candidate.jpg) |
| 22 | train | 3181.png | 0.475 | `candidate_shared_failure` | thin_component_frac, component_count, layout_cy_median | [panel](panels/22_train_3181_candidate.jpg) |
| 23 | train | 4181.png | 0.476 | `all_hard` | thin_component_frac, fg_mean, fg_bg_contrast | [panel](panels/23_train_4181_candidate.jpg) |
| 24 | train | 2123.png | 0.478 | `candidate_shared_failure` | compactness_median, component_area_max, component_area_cv | [panel](panels/24_train_2123_candidate.jpg) |
| 25 | train | 15317.png | 0.478 | `underreach_high_precision` | nearest_center_gap, thin_component_frac, layout_cy_median | [panel](panels/25_train_15317_candidate.jpg) |
| 26 | train | 5117.png | 0.486 | `candidate_fixable` | thin_component_frac, layout_cy_median, perimeter_area_median | [panel](panels/26_train_5117_candidate.jpg) |
| 27 | train | 2741.png | 0.488 | `under_separated_bridge` | nearest_center_gap, compactness_median, component_count | [panel](panels/27_train_2741_candidate.jpg) |
| 28 | train | 2152.png | 0.493 | `candidate_shared_failure` | component_area_iqr, nearest_center_gap, layout_width | [panel](panels/28_train_2152_candidate.jpg) |
| 29 | train | 2509.png | 0.494 | `under_separated_bridge` | compactness_median, aspect_median, component_count | [panel](panels/29_train_2509_candidate.jpg) |
| 30 | train | 5925.png | 0.497 | `underreach_high_precision` | layout_width, thin_component_frac, component_area_min | [panel](panels/30_train_5925_candidate.jpg) |
| 31 | val | 2044.png | 0.498 | `candidate_shared_failure` | layout_cy_median, layout_height, layout_cy_span | [panel](panels/31_val_2044_candidate.jpg) |
| 32 | train | 15177.png | 0.498 | `underreach_high_precision` | thin_component_frac, nearest_center_gap, component_area_cv | [panel](panels/32_train_15177_candidate.jpg) |
| 33 | train | 4339.png | 0.499 | `candidate_shared_failure` | aspect_median, component_area_iqr, layout_cy_median | [panel](panels/33_train_4339_candidate.jpg) |
| 34 | train | 3141.png | 0.500 | `overmask_low_precision` | compactness_median, component_area_iqr, component_area_median | [panel](panels/34_train_3141_candidate.jpg) |
| 35 | train | 3197.png | 0.512 | `overmask_low_precision` | compactness_median, layout_cy_median, nearest_center_gap | [panel](panels/35_train_3197_candidate.jpg) |
| 36 | train | 15339.png | 0.513 | `mixed_boundary_bridge` | layout_width, nearest_center_gap, bg_mean | [panel](panels/36_train_15339_candidate.jpg) |
| 37 | train | 2886.png | 0.513 | `overmask_low_precision` | compactness_median, aspect_median, layout_height | [panel](panels/37_train_2886_candidate.jpg) |
| 38 | val | 7269.png | 0.515 | `underreach_high_precision` | component_area_cv, layout_width, component_area_max | [panel](panels/38_val_7269_candidate.jpg) |
| 39 | train | 11779.png | 0.516 | `underreach_high_precision` | layout_width, nearest_center_gap, aspect_median | [panel](panels/39_train_11779_candidate.jpg) |
| 40 | train | 2906.png | 0.518 | `under_separated_bridge` | compactness_median, component_count, component_area_iqr | [panel](panels/40_train_2906_candidate.jpg) |
