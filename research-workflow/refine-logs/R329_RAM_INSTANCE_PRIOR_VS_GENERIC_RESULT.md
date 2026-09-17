# R329 RAM instance prior versus generic adapter

## Question

R329 tests whether the R328 gain is attributable to the shared per-bone
instance structure, rather than merely adding about 110k trainable parameters
behind the frozen R325 model.

## Protocol

- RAM-W600: 425 train / 69 validation; test not loaded.
- Frozen R325 native-resolution nnU-Net baseline, SHA256
  `88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9`.
- Native pixels; right/bottom padding only; no resize.
- Fixed threshold 0.5; no threshold search.
- Both arms use seed 3291, 25 epochs, learning rate 0.0002, identical
  checkpoint selection, and boundary weight 0.05.
- Shared per-instance prior: 108,914 trainable parameters.
- Generic 14-channel convolutional adapter: 110,440 trainable parameters
  (`+1.401%` versus the instance prior).
- No synthetic corruption is used.

## Validation results

| Configuration | Macro DSC | Macro IoU | Overlap DSC | Overlap IoU | NSD@2px | MSD px (lower better) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen R325 baseline | 0.978762507 | 0.958641648 | 0.872645023 | 0.774064197 | 0.780397702 | 1.968366591 |
| Generic matched adapter | 0.978773594 | 0.958661675 | 0.873190649 | 0.774923238 | 0.779106215 | 1.970348469 |
| Shared per-instance prior | **0.978902102** | **0.958909631** | **0.875346737** | **0.778325964** | **0.780808099** | **1.932129956** |

Shared per-instance prior minus frozen baseline:

- Macro DSC: `+0.000139594`.
- Macro IoU: `+0.000267982`.
- Overlap DSC: `+0.002701714`.
- Overlap IoU: `+0.004261768`.
- NSD@2px: `+0.000410397`.
- MSD: `-0.036236635 px` (better).

Generic adapter minus frozen baseline:

- Macro DSC: `+0.000011086`.
- Macro IoU: `+0.000020027`.
- Overlap DSC: `+0.000545627`.
- Overlap IoU: `+0.000859041`.
- NSD@2px: `-0.001291488` (worse).
- MSD: `+0.001981878 px` (worse).

Shared per-instance prior minus generic matched adapter:

- Macro DSC: `+0.000128508`.
- Macro IoU: `+0.000247955`.
- Overlap DSC: `+0.002156088`.
- Overlap IoU: `+0.003402726`.
- NSD@2px: `+0.001701885`.
- MSD: `-0.038218513 px` (better).

Both selected checkpoints are epoch 24, so the comparison is not caused by
one arm stopping substantially earlier.

## Interpretation and decision

Within this strictly paired single-seed validation experiment, the useful
effect is not explained by generic added capacity. The generic adapter has
slightly more parameters but is nearly neutral on region metrics and worsens
both boundary metrics. In contrast, the shared per-instance prior improves all
six tracked metrics over both the frozen baseline and the generic adapter.

This supports retaining the shared image-conditioned per-bone refinement
structure and boundary consistency, while continuing to exclude synthetic
corruption from the main route. The strongest improvements remain localized to
the overlap region (`+0.00270` DSC and `+0.00426` IoU versus baseline), which is
consistent with the intended mechanism rather than a broad capacity effect.

This is supporting evidence, not yet a final paper claim. R329 uses one seed
and one validation split. Statistical uncertainty and visual localization must
be checked before claiming model-independent or generalizable superiority.
Also, R329 is a paired rerun with deterministic DataLoader handling; its
absolute instance-prior score should not be substituted for the earlier R328
number without noting the protocol change.

## Subsequent official-metric audit

The corrected RAM paper-aligned per-instance/per-case audit supports a partial
go. The instance prior improves official overall DSC (`+0.000090`), VOE
(`-0.000168`) and MSD (`-0.034254 px`). On overlap regions it improves DSC
(`+0.002513`), NSD (`+0.000410`), VOE (`-0.003804`) and MSD (`-0.035070 px`).
Pair-intersection DSC/VOE/MSD also improve, but pair NSD is flat and RAVD
worsens. See `R329_RAM_OFFICIAL_METRICS_VALIDATION_RESULT.md`.

## Artifacts

- `outputs/ram_w600/r329_instance_prior_vs_generic/result.json`
- `outputs/ram_w600/r329_instance_prior_vs_generic/parameter_matching.json`
- checkpoints and histories in the same directory
- `outputs/bridge_logs/r329_instance_prior_vs_generic.log`
