# R327 RAM native-resolution instance-completion result

## Protocol

- Dataset: RAM-W600 official 425 train / 69 validation.
- Test split: not loaded.
- Baseline: frozen R325 native-resolution plain nnU-Net.
- Spatial handling: original 600/650/720 pixels, right/bottom padding only.
- Threshold: fixed at 0.5; no threshold search.
- Capacity control and completion-prior refiners use the same architecture,
  initialization, optimizer, seed, and checkpoint-selection rule.
- Completion mix: 50% mature-baseline probabilities, 30% algorithmic errors,
  20% clean identity samples.

## Full validation Oracle

Correcting only missing instance memberships inside true overlap changes:

- Macro DSC: `+0.008468926`.
- Macro IoU: `+0.016184926`.
- Overlap DSC: `+0.074370033`.
- Overlap IoU: `+0.125298234`.
- Overlap NSD@2px: `+0.118484055`.
- Overlap MSD: `-0.917656264 px`.

This verifies substantial remaining overlap-completion headroom in the mature
native-resolution baseline.

## Learned validation result

| Configuration | Macro DSC | Macro IoU | Overlap DSC | Overlap IoU | NSD@2px | MSD px (lower better) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen R325 baseline | 0.978762507 | 0.958641648 | 0.872645023 | 0.774064197 | **0.780397702** | 1.968366591 |
| Capacity-matched refiner | **0.978928089** | **0.958958685** | **0.875636539** | **0.778784324** | 0.779594776 | 1.938854535 |
| Error-recovery completion prior | 0.978882015 | 0.958868265 | 0.875169744 | 0.778046144 | 0.778091126 | **1.933446710** |

Completion prior minus frozen baseline:

- Macro DSC: `+0.000119507`.
- Macro IoU: `+0.000226617`.
- Overlap DSC: `+0.002524722`.
- Overlap IoU: `+0.003981947`.
- Overlap NSD@2px: `-0.002306577`.
- Overlap MSD: `-0.034919882 px`.

Completion prior minus capacity control:

- Macro DSC: `-0.000046074`.
- Macro IoU: `-0.000090420`.
- Overlap DSC: `-0.000466795`.
- Overlap IoU: `-0.000738180`.
- Overlap NSD@2px: `-0.001503651`.
- Overlap MSD: `-0.005407826 px` (better).

## Decision

The shared per-instance native-resolution refiner is mechanism-positive: it
produces materially larger overlap gains than R326 while preserving and
improving Macro DSC/IoU. The current algorithmic-corruption prior is not yet
independently supported. Relative to the capacity control it improves only
MSD, while area metrics and NSD are worse.

The likely problem is corruption-distribution mismatch. The 30% synthetic arm
mixes overlap dropout, boundary dropout, and false membership. For RAM, false
membership and broad boundary corruption dilute the main real error mode:
missing secondary-bone membership in genuine projection overlap. The next
controlled variant should retain this refiner and initialization, but replace
generic corruption with baseline-error-shaped, FN-dominant overlap dropout
and reduce the synthetic fraction rather than add more capacity.

## Artifacts

- `outputs/ram_w600/r327_native_instance_completion_prior/result.json`
- `outputs/ram_w600/r327_native_instance_completion_prior/oracle.json`
- capacity and completion checkpoints/histories in the same directory
- `outputs/bridge_logs/r327_native_instance_completion_prior.log`
