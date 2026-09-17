# R330 RAM pair-surface and volume-prior result

## Protocol

- RAM-W600 validation only; test not loaded.
- Fixed threshold 0.5; no threshold search.
- Native resolution with right/bottom padding only.
- Frozen R325 baseline; both variants initialized from the same R329 instance-prior checkpoint.
- `surface003_volume010`: soft NSD weight 0.03, instance-volume weight 0.005, pair-volume weight 0.01.
- `surface006_volume020`: soft NSD weight 0.06, instance-volume weight 0.01, pair-volume weight 0.02.
- Checkpoint gate requires pooled Macro DSC and IoU not below R325; RAM official metrics are then audited per instance/case.

## Official overall-instance metrics

| Method | DSC up | NSD@2px up | VOE down | MSD px down | RAVD down |
|---|---:|---:|---:|---:|---:|
| R325 | 0.978858 | 0.887262 | 0.040899 | 1.169285 | 0.020263 |
| R329 | 0.978947 | 0.887175 | 0.040732 | 1.135031 | 0.020588 |
| R330 weak | **0.979150** | **0.890012** | **0.040349** | **1.110572** | **0.020163** |
| R330 strong | 0.979046 | 0.889211 | 0.040543 | 1.122416 | 0.020254 |

R330 weak minus R325: DSC `+0.000292`, NSD `+0.002750`, VOE
`-0.000550`, MSD `-0.058714 px`, RAVD `-0.000100`.

## Official overlap-region metrics

| Method | DSC up | NSD@2px up | VOE down | MSD px down | RAVD down |
|---|---:|---:|---:|---:|---:|
| R325 | 0.873731 | 0.780398 | 0.222655 | 1.946378 | **0.051058** |
| R329 | **0.876244** | 0.780808 | **0.218851** | 1.911308 | 0.053971 |
| R330 weak | 0.875785 | **0.784118** | 0.219567 | **1.859983** | 0.051602 |
| R330 strong | 0.875688 | 0.783327 | 0.219682 | 1.891835 | 0.050210 |

R330 weak minus R325: DSC `+0.002054`, NSD `+0.003720`, VOE
`-0.003088`, MSD `-0.086395 px`, RAVD `+0.000544`.

## Official overlap pair-intersection metrics

| Method | DSC up | NSD@2px up | VOE down | MSD px down | RAVD down |
|---|---:|---:|---:|---:|---:|
| R325 | 0.833630 | 0.782834 | 0.261175 | 1.742422 | 0.666188 |
| R329 | **0.836789** | 0.782790 | **0.256973** | 1.696062 | 0.712233 |
| R330 weak | 0.834990 | 0.784964 | 0.258217 | **1.678961** | **0.654402** |
| R330 strong | 0.835477 | **0.785173** | 0.258288 | 1.694969 | 0.678288 |

R330 weak minus R325: DSC `+0.001360`, NSD `+0.002130`, VOE
`-0.002958`, MSD `-0.063461 px`, RAVD `-0.011786`. MSD failure rate is unchanged.

## Decision

The weak R330 configuration is the best balanced result and passes the intended
mechanism test. It is the first configuration in this chain to improve all RAM
official overall-instance metrics and all official pair-intersection metrics
against R325. The surface surrogate fixes R329's flat NSD, while pair-volume
supervision reverses R329's pair RAVD degradation.

The stronger weights do not improve the overall balance. They produce the best
pair NSD and overlap-region RAVD but lose overall DSC/NSD/MSD and pair RAVD
relative to the weak configuration. This supports a moderate constraint rather
than stronger intervention.

R330 does not dominate R329 on every overlap-region metric: R329 retains a
small advantage in overlap DSC (`0.000459`) and VOE (`0.000716`), while R330
weak is better in overlap NSD (`+0.003310`), MSD (`-0.051324 px`) and RAVD
(`-0.002370`). Therefore R330 should be treated as the balanced boundary/volume
candidate, not as an across-the-board replacement until multi-seed validation.

## Artifacts

- `outputs/ram_w600/r330_pair_surface_volume_prior/result.json`
- `outputs/ram_w600/r330_pair_surface_volume_prior/surface003_volume010_best.pth`
- `outputs/ram_w600/r330_pair_surface_volume_prior/surface006_volume020_best.pth`
- `outputs/bridge_logs/r330_pair_surface_volume.log`
