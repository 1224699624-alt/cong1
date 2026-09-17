# R331 R330 weak-configuration five-seed stability result

## Protocol

- RAM-W600 validation only; test never loaded.
- Seeds: 1024, 2025, 3407, 4096, 5214.
- Each seed starts from the same frozen R325 baseline and the same R329
  instance-refiner checkpoint; only the R330 refinement training RNG changes.
- Fixed R330 weak configuration: surface 0.03, instance-volume 0.005,
  pair-volume 0.01.
- Native pixels; right/bottom padding only; threshold 0.5.
- Checkpoint gate retains Macro DSC/IoU non-degradation versus R325.

## Five-seed official metrics

`↑` means higher is better; `↓` means lower is better. Values are mean ±
sample standard deviation. The arrow in the change column shows how R330 moved
relative to R325, while the final column states whether that movement is
beneficial.

### Overall instance

| Metric / preferred direction | R325 | R330 mean ± std | Actual change | Assessment | Better seeds |
|---|---:|---:|---:|---|---:|
| DSC ↑ | 0.978858 | **0.979117 ± 0.000049** | ↑ +0.000260 | **Beneficial** | 5/5 |
| NSD@2px ↑ | 0.887262 | **0.889882 ± 0.000483** | ↑ +0.002620 | **Beneficial** | 5/5 |
| VOE ↓ | 0.040899 | **0.040409 ± 0.000090** | ↓ -0.000491 | **Beneficial** | 5/5 |
| MSD ↓ | 1.169285 | **1.112718 ± 0.006095** | ↓ -0.056567 px | **Beneficial** | 5/5 |
| RAVD ↓ | 0.020263 | **0.020201 ± 0.000028** | ↓ -0.000062 | **Beneficial** | 5/5 |
| MSD fail rate ↓ | 0.000000 | 0.000000 ± 0.000000 | → 0.000000 | Neutral | 0/5 |

### Overlap region

| Metric / preferred direction | R325 | R330 mean ± std | Actual change | Assessment | Better seeds |
|---|---:|---:|---:|---|---:|
| DSC ↑ | 0.873731 | **0.875311 ± 0.000315** | ↑ +0.001581 | **Beneficial** | 5/5 |
| NSD@2px ↑ | 0.780398 | **0.784152 ± 0.000535** | ↑ +0.003755 | **Beneficial** | 5/5 |
| VOE ↓ | 0.222655 | **0.220282 ± 0.000487** | ↓ -0.002374 | **Beneficial** | 5/5 |
| MSD ↓ | 1.946378 | **1.862337 ± 0.013665** | ↓ -0.084041 px | **Beneficial** | 5/5 |
| RAVD ↓ | **0.051058** | 0.052741 ± 0.001244 | ↑ +0.001682 | **Unfavorable** | 0/5 |
| MSD fail rate ↓ | 0.000000 | 0.000000 ± 0.000000 | → 0.000000 | Neutral | 0/5 |

### Pair-intersection region

| Metric / preferred direction | R325 | R330 mean ± std | Actual change | Assessment | Better seeds |
|---|---:|---:|---:|---|---:|
| DSC ↑ | 0.833630 | **0.834603 ± 0.000457** | ↑ +0.000972 | **Beneficial** | 5/5 |
| NSD@2px ↑ | 0.782834 | **0.785243 ± 0.000337** | ↑ +0.002409 | **Beneficial** | 5/5 |
| VOE ↓ | 0.261175 | **0.258864 ± 0.000608** | ↓ -0.002311 | **Beneficial** | 5/5 |
| MSD ↓ | 1.742422 | **1.682623 ± 0.004975** | ↓ -0.059799 px | **Beneficial** | 5/5 |
| RAVD ↓ | 0.666188 | **0.651639 ± 0.010190** | ↓ -0.014549 | **Beneficial overall** | 4/5 |
| MSD fail rate ↓ | **0.001992** | 0.002789 ± 0.000445 | ↑ +0.000797 | **Unfavorable** | 0/5 |

The overlap-region RAVD remains the only consistently unfavorable metric: all
five seeds are slightly above R325 (mean 0.052741 versus 0.051058). Pair MSD
failure rate is also not improved (mean 0.002789 versus 0.001992), although
the absolute failure rate remains below 0.3%.

## Interpretation

The core R330 conclusion is stable across seeds. Improvements are not caused by
one favorable initialization: every seed improves the official overall and
overlap DSC/NSD/VOE/MSD metrics. The standard deviations are small relative to
the gains, especially for MSD and VOE. This supports the claim that moderate
surface-plus-volume supervision is more reliable than the stronger weight
variant tested in R330.

The remaining RAVD issue is localized to the union overlap region rather than
the overall volume metric. It should be treated as a known limitation and
addressed with pair-specific volume calibration or uncertainty gating before
any test-set promotion.

## Visual protocol

Six validation cases were selected by the R325 baseline's lowest overlap-region
DSC, without using R329/R330 gains. Each panel contains the original image,
ground truth, R325, R329, and R330, plus a common yellow-box local crop. This is
a difficulty-based qualitative audit, not test-set model selection.

## Artifacts

- `outputs/ram_w600/r331_r330_five_seed/result.json`
- seed checkpoints and histories in the same directory
- `outputs/visualizations/r331_ram_r325_r329_r330/r331_baseline_hard_contact_sheet.png`
- six per-case comparison PNGs in the same directory
- `outputs/visualizations/r331_ram_r325_r329_r330/r331_multiseed_stability.pdf`
- `outputs/visualizations/r331_ram_r325_r329_r330/r331_multiseed_stability.png`
