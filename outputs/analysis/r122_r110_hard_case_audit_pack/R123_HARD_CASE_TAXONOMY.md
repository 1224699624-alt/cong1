# R123 Hard-Case Taxonomy

This is an auto-initialized taxonomy for the R122 visual audit pack. Treat `auto_tags` as a triage aid, not final labels.

## Tag Counts

- `candidate_fixable`: 7
- `overmask_low_precision`: 7
- `candidate_shared_failure`: 6
- `under_separated_bridge`: 6
- `severe_boundary_shift`: 6
- `underreach_high_precision`: 4
- `mixed_boundary_bridge`: 2

## Decision Gate

- If most manually confirmed cases are `candidate_fixable`, curate matching train/val examples before a new learner.
- If many cases are `candidate_shared_failure`, stop candidate readout experiments and revise data/label protocol.
- If `under_separated_bridge` dominates, prioritize hard-case training data or explicit bridge labels, not threshold sweeps.

## Cases

| Rank | Image | Dice | Auto Tags | Suggested Action | Panel |
| ---: | --- | ---: | --- | --- | --- |
| 1 | 4818.png | 0.8530 | `candidate_shared_failure` | visual/label review or collect matching hard train examples; do not use candidate readout alone | [panel](panels/01_4818_audit.jpg) |
| 2 | 12936.png | 0.8592 | `candidate_fixable` | candidate-guided training only if matching train/val examples can be curated | [panel](panels/02_12936_audit.jpg) |
| 3 | 3665.png | 0.8760 | `under_separated_bridge` | inspect bridge/low-contrast pattern; consider hard-case data curation before new model | [panel](panels/03_3665_audit.jpg) |
| 4 | 9194.png | 0.8779 | `underreach_high_precision` | inspect label extent; consider boundary/recall protocol review | [panel](panels/04_9194_audit.jpg) |
| 5 | 2136.png | 0.8797 | `mixed_boundary_bridge` | manual review before experiment | [panel](panels/05_2136_audit.jpg) |
| 6 | 8435.png | 0.8849 | `underreach_high_precision, candidate_fixable` | candidate-guided training only if matching train/val examples can be curated | [panel](panels/06_8435_audit.jpg) |
| 7 | 7484.png | 0.8866 | `candidate_fixable` | candidate-guided training only if matching train/val examples can be curated | [panel](panels/07_7484_audit.jpg) |
| 8 | 1897.png | 0.8897 | `severe_boundary_shift, candidate_shared_failure` | visual/label review or collect matching hard train examples; do not use candidate readout alone | [panel](panels/08_1897_audit.jpg) |
| 9 | 2050.png | 0.8901 | `under_separated_bridge, overmask_low_precision, candidate_shared_failure` | visual/label review or collect matching hard train examples; do not use candidate readout alone | [panel](panels/09_2050_audit.jpg) |
| 10 | 2075.png | 0.8923 | `under_separated_bridge, overmask_low_precision, candidate_shared_failure` | visual/label review or collect matching hard train examples; do not use candidate readout alone | [panel](panels/10_2075_audit.jpg) |
| 11 | 4099.png | 0.8957 | `severe_boundary_shift` | manual review before experiment | [panel](panels/11_4099_audit.jpg) |
| 12 | 4078.png | 0.8970 | `underreach_high_precision, severe_boundary_shift` | inspect label extent; consider boundary/recall protocol review | [panel](panels/12_4078_audit.jpg) |
| 13 | 4319.png | 0.8988 | `overmask_low_precision, candidate_shared_failure` | visual/label review or collect matching hard train examples; do not use candidate readout alone | [panel](panels/13_4319_audit.jpg) |
| 14 | 1468.png | 0.8997 | `under_separated_bridge, overmask_low_precision` | inspect bridge/low-contrast pattern; consider hard-case data curation before new model | [panel](panels/14_1468_audit.jpg) |
| 15 | 3900.png | 0.8997 | `severe_boundary_shift` | manual review before experiment | [panel](panels/15_3900_audit.jpg) |
| 16 | 2982.png | 0.9016 | `candidate_fixable` | candidate-guided training only if matching train/val examples can be curated | [panel](panels/16_2982_audit.jpg) |
| 17 | 2495.png | 0.9018 | `candidate_shared_failure` | visual/label review or collect matching hard train examples; do not use candidate readout alone | [panel](panels/17_2495_audit.jpg) |
| 18 | 1784.png | 0.9027 | `candidate_fixable` | candidate-guided training only if matching train/val examples can be curated | [panel](panels/18_1784_audit.jpg) |
| 19 | 1789.png | 0.9052 | `severe_boundary_shift, candidate_fixable` | candidate-guided training only if matching train/val examples can be curated | [panel](panels/19_1789_audit.jpg) |
| 20 | 7397.png | 0.9054 | `mixed_boundary_bridge` | manual review before experiment | [panel](panels/20_7397_audit.jpg) |
| 21 | 1891.png | 0.9103 | `overmask_low_precision, severe_boundary_shift` | inspect bridge/low-contrast pattern; consider hard-case data curation before new model | [panel](panels/21_1891_audit.jpg) |
| 22 | 2168.png | 0.9147 | `under_separated_bridge, overmask_low_precision` | inspect bridge/low-contrast pattern; consider hard-case data curation before new model | [panel](panels/22_2168_audit.jpg) |
| 23 | 3660.png | 0.9065 | `under_separated_bridge, overmask_low_precision` | inspect bridge/low-contrast pattern; consider hard-case data curation before new model | [panel](panels/23_3660_audit.jpg) |
| 24 | 15578.png | 0.9059 | `underreach_high_precision, candidate_fixable` | candidate-guided training only if matching train/val examples can be curated | [panel](panels/24_15578_audit.jpg) |
