# R127 Hard-Case Protocol Audit

## Raw Numbers

- train total: `875`
- val total: `96`
- R125 hard-train: `64`
- R125 hard-val-audit: `6`
- severe hard-pool robust outliers (top3 z >= 3.5): `0`

## R125 Prototype Counts

- `underreach_high_precision`: `18`
- `under_separated_bridge`: `16`
- `candidate_shared_failure`: `9`
- `overmask_low_precision`: `7`
- `severe_boundary_shift`: `5`
- `mixed_boundary_bridge`: `4`
- `all_hard`: `4`
- `candidate_fixable`: `1`

## R126 On R122 Hard Cases

- overlap count: `24`
- mean Dice: `0.868973`
- mean Boundary IoU: `0.151140`
- mean component-count error: `3.416667`
- false-bridge mean: `0.833333`

## Decision

- simple resampling supported: `False`
- broad label filtering supported: `False`
- manual label/protocol review supported: `True`
- next GPU without new labels supported: `False`

## Rationale

- R126 hard-case sampler underperformed R117/R110, so the current hard pool is not sufficient as weighting signal.
- R125 hard-train prototypes are dominated by shared-failure, bridge, and underreach-like cases rather than candidate-fixable cases.
- Use severe feature outliers and hard-case tags to prioritize manual label/protocol review before another training run.

## Top Severe Outliers

