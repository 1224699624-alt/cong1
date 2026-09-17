# R072 Data/Layout Audit Summary

Date: 2026-06-26

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Inputs

R072 used only existing clean-test-v2 ground truth and predictions. It did not use the new reannotated test split.

Audited systems:

- R038 current best anchor
- R068 instance-separation direct segmenter
- R069 R038 + separation veto
- R071 radial instance segmenter

Remote artifacts:

- `outputs/analysis/r072_clean_test_v2_consensus_suspect_labels.json`
- `outputs/analysis/r072_r038_layout_failure_clean_test_v2.json`
- `outputs/analysis/r072_r071_layout_failure_clean_test_v2.json`

## Findings

### Consensus Suspect Labels

Only `5` images met the strong high-consensus / low-GT criterion.

Top consensus suspects:

| Image | Mean Dice to GT | Mean Pairwise Dice | Consensus Dice |
| --- | ---: | ---: | ---: |
| `4818.png` | `0.8432` | `0.9454` | `0.8522` |
| `1897.png` | `0.8829` | `0.9523` | `0.8855` |
| `4078.png` | `0.8964` | `0.9623` | `0.8926` |
| `4319.png` | `0.8752` | `0.9513` | `0.8925` |
| `2050.png` | `0.8821` | `0.9569` | `0.8951` |

Interpretation: there are likely a few label-definition or hard-boundary cases, but not enough to explain the full `+0.017409` target gap.

### R038 Layout Failure Profile

R038 clean-test-v2:

- Dice `0.914357`
- Precision `0.899054`
- Recall `0.931483`
- Boundary IoU `0.243176`
- Mean component-count error `3.160494`
- `16/81` images have Dice `< 0.90`
- `29/81` images have Boundary IoU `< 0.20`
- `24/81` images are bridge-like by the audit heuristic
- `5/81` images are missing-component-like
- `10/81` images are overmask-like

Worst R038 images by Dice:

`4818.png`, `12936.png`, `9194.png`, `2136.png`, `3665.png`, `7484.png`, `1897.png`, `2050.png`, `4319.png`, `2075.png`.

### R071 Comparison

R071 has lower component-count error (`1.93`) than R038 but worse Dice and much worse Boundary IoU, meaning component counting alone is not the bottleneck. The missing piece is not merely getting the number of instances closer; the method needs boundary-aware layout constraints that preserve R038's strong region quality.

## R073 Speed Note

A watershed-cut layout repair diagnostic was started but stopped because even a 4-config val grid was too slow for a quick non-GPU gate. The slow diagnostic produced no result and should not be treated as evidence for or against layout repair.

## Decision

Do not pursue a broad data deletion/filtering run yet. R072 does not show enough widespread suspect-label evidence.

The next useful step is R074: a lightweight anatomy-layout constraint around R038 that is faster than watershed and explicitly targets bridge-like cases while preserving recall. It should be selected on original val using an anchor proxy and applied once to clean-test-v2. If this also fails, the next larger move should be an isolated train/val label/layout audit dataset variant, not another direct architecture.
