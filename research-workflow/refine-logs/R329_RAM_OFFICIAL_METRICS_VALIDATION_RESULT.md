# R329 RAM official-metric validation audit

## Protocol

- RAM-W600 validation only: 69 images / 966 bone-instance evaluations.
- RAM test was not loaded or used.
- Threshold fixed at 0.5; no threshold search.
- Native pixels with right/bottom padding only.
- Same frozen R325 baseline and R329 checkpoints.
- RAM metrics: per-instance/per-case DSC, NSD@2px, VOE, symmetric MSD,
  MSD Fail Rate, and RAVD.
- Pair evaluation uses the top 15 train-set overlap pairs and excludes empty-GT
  pair/case combinations.

## Overall instance metrics

| Method | DSC up | NSD@2px up | VOE down | MSD px down | Fail rate down | RAVD down |
|---|---:|---:|---:|---:|---:|---:|
| R325 baseline | 0.978858 | **0.887262** | 0.040899 | 1.169285 | 0.000000 | **0.020263** |
| Generic adapter | 0.978847 | 0.886158 | 0.040925 | 1.170207 | 0.000000 | 0.020209 |
| Instance prior | **0.978947** | 0.887175 | **0.040732** | **1.135031** | 0.000000 | 0.020588 |

Instance prior minus baseline: DSC `+0.000090`, NSD `-0.000087`, VOE
`-0.000168` (better), MSD `-0.034254 px` (better), RAVD `+0.000325`
(worse).

## Overlap-region metrics

| Method | DSC up | NSD@2px up | VOE down | MSD px down | Fail rate down | RAVD down |
|---|---:|---:|---:|---:|---:|---:|
| R325 baseline | 0.873731 | 0.780398 | 0.222655 | 1.946378 | 0.000000 | **0.051058** |
| Generic adapter | 0.874211 | 0.779106 | 0.221919 | 1.947478 | 0.000000 | 0.050196 |
| Instance prior | **0.876244** | **0.780808** | **0.218851** | **1.911308** | 0.000000 | 0.053971 |

Instance prior minus baseline: DSC `+0.002513`, NSD `+0.000410`, VOE
`-0.003804` (better), MSD `-0.035070 px` (better), RAVD `+0.002913`
(worse).

## Pair-intersection metrics

| Method | DSC up | NSD@2px up | VOE down | MSD px down | Fail rate down | RAVD down |
|---|---:|---:|---:|---:|---:|---:|
| R325 baseline | 0.833630 | **0.782834** | 0.261175 | 1.742422 | 0.001992 | **0.666188** |
| Generic adapter | 0.834147 | 0.781776 | 0.260523 | 1.737133 | 0.001992 | 0.671679 |
| Instance prior | **0.836789** | 0.782790 | **0.256973** | **1.696062** | 0.001992 | 0.712233 |

Instance prior minus baseline: DSC `+0.003159`, NSD `-0.000044`, VOE
`-0.004202` (better), MSD `-0.046360 px` (better), fail rate unchanged,
and RAVD `+0.046045` (worse).

## Decision

The corrected official audit supports a partial-go, not a full promotion.
The instance prior improves overall DSC, VOE and MSD, and produces stronger
improvements on overlap-region and pair-intersection DSC/VOE/MSD. It also beats
the parameter-matched generic adapter on the intended difficult-region metrics.

However, overall NSD is essentially flat/slightly worse, pair NSD is flat, and
RAVD worsens, particularly for pair intersections. The module is improving
where predicted overlap pixels are placed while introducing a volume-bias error
for some pairs. The next revision should target pair-wise volume conservation
and surface agreement; merely increasing the residual magnitude is unsafe.

No comparison against the RAM paper's test numbers is valid yet because this
audit is validation-only, native-resolution, and single-seed.

## Artifacts

- `outputs/ram_w600/r329_official_metrics_val/result.json`
- `outputs/bridge_logs/r329_official_metrics_val.log`
- evaluator: `scripts/evaluate_r329_ram_official_metrics_val.py`
