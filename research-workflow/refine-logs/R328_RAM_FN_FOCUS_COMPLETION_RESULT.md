# R328 RAM FN-focused completion result

## Protocol

- RAM-W600: 425 train / 69 validation; test not loaded.
- Frozen R325 native-resolution nnU-Net baseline.
- Original pixels with right/bottom padding only.
- Fixed threshold 0.5; no threshold search.
- Same per-instance refiner initialization for both arms.
- Both arms use boundary consistency weight 0.05.
- Prior arm mix: 70% baseline predictions, 12% FN-focused overlap
  corruption, 18% clean identity samples.

## Validation

| Configuration | Macro DSC | Macro IoU | Overlap DSC | Overlap IoU | NSD@2px | MSD px (lower better) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen R325 baseline | 0.978762507 | 0.958641648 | 0.872645023 | 0.774064197 | 0.780397702 | 1.968366591 |
| Instance refiner + boundary loss, no synthetic corruption | **0.979110897** | **0.959307849** | **0.876587518** | **0.780290080** | **0.783637032** | **1.887910685** |
| FN-focused synthetic completion prior | 0.978983223 | 0.959062099 | 0.875077789 | 0.777900801 | 0.781443455 | 1.971622023 |

No-synthetic instance refiner minus frozen baseline:

- Macro DSC: `+0.000348389`.
- Macro IoU: `+0.000666201`.
- Overlap DSC: `+0.003942495`.
- Overlap IoU: `+0.006225883`.
- NSD@2px: `+0.003239330`.
- MSD: `-0.080455906 px`.

FN-focused prior minus frozen baseline:

- Macro DSC: `+0.000220716`.
- Macro IoU: `+0.000420451`.
- Overlap DSC: `+0.002432767`.
- Overlap IoU: `+0.003836604`.
- NSD@2px: `+0.001045753`.
- MSD: `+0.003255432 px` (worse).

FN-focused prior minus the paired no-synthetic refiner:

- Macro DSC: `-0.000127673`.
- Macro IoU: `-0.000245750`.
- Overlap DSC: `-0.001509728`.
- Overlap IoU: `-0.002389279`.
- NSD@2px: `-0.002193577`.
- MSD: `+0.083711338 px` (worse).

## Decision

The boundary-consistent, image-conditioned shared per-instance refiner is the
first RAM configuration in this chain to improve every tracked metric over the
mature native-resolution baseline. The FN-focused synthetic corruption is
not supported: even after narrowing and reducing its fraction, it is worse
than the paired no-synthetic arm on every metric.

This localizes the useful mechanism to training on the mature model's real
prediction residuals plus boundary consistency. Synthetic mask corruption
introduces a prediction-distribution mismatch and should be removed from the
main route. A future paper claim should call the no-synthetic module a learned
image-conditioned instance refinement prior, and separately verify it against
a genuinely generic parameter-matched adapter; it should not claim that
synthetic error recovery caused the R328 gains.

## Artifacts

- `outputs/ram_w600/r328_fn_focus_completion/result.json`
- checkpoints and histories in the same directory
- `outputs/bridge_logs/r328_fn_focus_completion.log`
