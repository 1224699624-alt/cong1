# R163 Reannotated Label-Shift Decision

## Status

DONE_DIAGNOSTIC.

R163 was a non-GPU follow-up to R162. It audited whether the train/val-only reannotated dataset is a small correction of the original labels or a materially different label protocol.

Evidence file:

`outputs/analysis/r163_reannotated_pair_shift.json`

## Raw Findings

Paired original-vs-reannotated train labels:

- Shared train labels: `875`
- Extra reannotated train labels: `381`
- Mean paired label Dice: `0.862338`
- Median paired label Dice: `0.905921`
- Q10 paired label Dice: `0.685225`
- Mean foreground-area ratio: `1.028503`
- Mean component-count delta: `-0.329143`

Paired original-vs-reannotated val labels:

- Shared val labels: `96`
- Extra reannotated val labels: `0`
- Mean paired label Dice: `0.945487`
- Median paired label Dice: `1.0`
- Q10 paired label Dice: `0.849205`
- Mean foreground-area ratio: `1.025838`
- Mean component-count delta: `0.291667`

Distribution means:

- Original train foreground fraction: `0.029734`; components: `23.432`
- Reannotated train foreground fraction: `0.029261`; components: `22.922`
- Original val foreground fraction: `0.026533`; components: `23.229`
- Reannotated val foreground fraction: `0.026032`; components: `23.521`
- clean-test-v2 foreground fraction: `0.030819`; components: `23.568`

Notable train outliers:

- `15047.png`, `15057.png`, `15087.png`, `15113.png`, and `15118.png` have reannotated foreground fraction `0.0` despite non-empty original labels.
- `2118.png`, `2101.png`, `2096.png`, `2108.png`, and `2099.png` have reannotated foreground fractions roughly `3x` to `5x` the original and low paired label Dice.

## Interpretation

R162 did not simply underperform because of a hyperparameter miss. The reannotated train split contains a mixed protocol:

- many labels are close to original,
- some same-name labels are empty or near-empty,
- some labels are much larger than original,
- and the train split has `381` extra cases while val has no extras.

The validation side is much closer to the original protocol than the training side. This train/val protocol mismatch is consistent with R162's weak best validation Dice `0.812450`, later `nan` instability, and final clean-test-v2 Dice `0.872626`.

## Decision

Do not launch R163/R164 as another full GPU retrain on the raw reannotated variant.

The next useful ARIS branch should be a gated data-cleaning variant, not another architecture sweep:

1. Build a filtered reannotated train variant that excludes empty labels and extreme paired-shift outliers.
2. Keep reannotated val only if its protocol remains close to clean-test-v2 and original val.
3. Use clean-test-v2/test only for final success evaluation.
4. Run a small path/gate first before any full GPU job.

Suggested next experiment:

`R164_reannotated_filtered_trainval_variant`

Gate criteria before full training:

- no empty train labels,
- paired train label Dice outliers below a conservative threshold are excluded,
- extreme foreground-area ratios are excluded,
- split counts and paired image/label names are verified,
- launcher explicitly avoids reannotated test and uses clean-test-v2/test only for success.
