# R326 lambda=0.001 result

## Protocol

- Server: westb port 29459, RTX 4090 D.
- Dataset: RAM-W600 official 425 train / 69 validation.
- Test split was not loaded; threshold remained fixed at 0.5.
- Baseline, initialization, seed, epochs, learning rate, preprocessing, and
  checkpoint-selection tuple were unchanged from R326.
- The only method change was `projection_weight: 0.00025 -> 0.001`.

## Validation comparison

| Projection weight | Macro DSC | Macro IoU | Overlap DSC | Overlap IoU | Overlap NSD@2px | Overlap MSD px (lower better) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (capacity-matched seg-only) | 0.978159189 | 0.957466602 | **0.873004960** | **0.774630941** | **0.879019930** | 1.125427198 |
| 0.00025 | **0.978162825** | **0.957473636** | 0.872921727 | 0.774499889 | 0.878769003 | **1.124241192** |
| 0.001 | 0.978149176 | 0.957447350 | 0.872879039 | 0.774432681 | 0.878678377 | 1.126928509 |

Relative to `lambda=0.00025`, `lambda=0.001` changed:

- Macro DSC: `-0.000013649`.
- Macro IoU: `-0.000026286`.
- Overlap DSC: `-0.000042689`.
- Overlap IoU: `-0.000067207`.
- Overlap NSD@2px: `-0.000090626`.
- Overlap MSD: `+0.002687317 px` (worse).

Relative to the frozen mature baseline, `lambda=0.001` still produced small
positive deltas in Macro DSC/IoU, Overlap DSC/IoU/NSD, and a lower MSD. It was,
however, worse than the capacity-matched segmentation-only adapter on every
listed metric.

## Interpretation

Increasing the projection coefficient by four did not expose a hidden stronger
overlap benefit. The best projection checkpoint moved from epoch 22 to epoch
21 and its training projection loss decreased, which confirms that the stronger
coefficient did make the adapter fit the pseudo radiographic layers more
aggressively. The segmentation and overlap metrics simultaneously worsened.
This is evidence of an objective conflict: the equal optical-density pseudo
layer target is reconstructable but does not contain sufficiently identifiable
per-bone attenuation information.

Decision: do not increase this pseudo-layer coefficient further. A stronger
follow-up should improve layer supervision or gate the projection loss to
reliable overlap regions rather than continue a scalar-weight sweep.

## Artifacts

- `outputs/ram_w600/r326_layer_projection_prior_lambda001/result.json`
- adapter checkpoints and complete histories in the same directory
- `outputs/bridge_logs/r326_layer_projection_prior_lambda001.log`
