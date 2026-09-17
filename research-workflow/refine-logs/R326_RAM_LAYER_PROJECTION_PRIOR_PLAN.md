# R326 RAM layer-projection prior experiment

## Objective

Test whether the bone-layer separation and radiographic reconstruction ideas
from BLS-GAN (AAAI 2025) and LSN (ACM MM 2025) can improve RAM-W600
multi-label wrist segmentation without reducing macro DSC/IoU.

## Dataset and isolation

- Dataset: RAM-W600 only.
- Split: official 425 train / 69 validation.
- Test split: locked and unused.
- Baseline: mature R285 14-label nnU-Net checkpoint, frozen.
- Output: `outputs/ram_w600/r326_layer_projection_prior_batched`.
- Threshold: fixed at 0.5; no threshold search.

## Paired arms

1. Frozen mature R285 baseline.
2. Frozen baseline plus a small segmentation-only post-logit adapter.
3. The identical adapter plus layer/projection supervision.

The second arm controls for adapter capacity so an improvement cannot be
attributed merely to adding parameters.

## RAM adaptation of the papers

- BLS-GAN/LSN accept a joint image and bone masks and predict radiographic bone
  layers. R326 replaces inference-time GT masks with the mature nnU-Net's 14
  probability maps.
- The two-bone output is generalized to fourteen independent bone layers.
- A transmission-style reconstruction is retained:
  `R = 1 - product_i(1 - L_i) * (1 - L_soft)`.
- Because RAM has masks but no separated layer-image GT, training constructs
  pseudo layer targets by splitting optical density equally across all GT bone
  memberships at each pixel. This is an explicit approximation and not claimed
  to recover true physical attenuation.
- The adapter remains active at inference, but uses only the radiograph and
  baseline probabilities; validation/test GT is never an inference input.

## Metrics and decision

- Macro DSC and Macro IoU are hard non-degradation metrics.
- Overlap DSC/IoU, Overlap NSD@2px, and Overlap MSD are the target metrics.
- The projection arm must first beat both macro metrics of the paired
  segmentation-only adapter before an overlap gain is considered a go.

## Training-only gradient calibration

- Initial one-batch native adapter gradient norm: `0.000712566`.
- Initial projection gradient norm: `0.900192`.
- An uncalibrated coefficient of `0.05` would make the prior/native gradient
  ratio about `63.17`, which would overwhelm the mature segmentation signal.
- R326 fixes `projection_weight=0.00025`, corresponding to an initial ratio of
  approximately `0.316`. This value is derived from a RAM training batch only;
  validation and test data are not used for coefficient selection.

## Source-code audit note

The public LSN `random_movement` function currently has its affine movement
block commented out and falls back to mask multiplication. R326 therefore does
not claim to reproduce LSN random shifting and does not silently depend on that
inactive implementation.

## Runtime adjustment

The first intact epoch with batch size 1 took 123.997 seconds while using only
about 1.2 GiB of the RTX 4090. Its record is preserved under the unbatched
output folder. The definitive run uses batch size 4 and four data-loader
workers in a new isolated output folder; method, split, seed, and selection
rule are unchanged.
