# R326 RAM layer-projection prior result

## Status

- Completed normally on 2026-08-05.
- Server: westb port 29459, RTX 4090 D.
- Dataset: RAM-W600 official 425 train / 69 validation.
- Test split: not loaded.
- Threshold: fixed at 0.5; no threshold search.
- Baseline SHA256: `3e42fbeb763f6e1fafc80e550c1c630b7fd599ff8cb85f7462ff8b5926475d01`.
- Output size: approximately 356 KiB; server data disk remained at 89% usage.

## Paper and code findings

### BLS-GAN

- The generator receives the radiograph concatenated with bone masks and
  predicts mask-supported bone-layer images.
- A segmentation-based multi-channel supervisor distinguishes overlap,
  non-overlap, and generated-layer authenticity.
- A learned scalar correction parameter modifies the transmission-style
  reconstruction in overlap regions.
- Synthetic non-overlap-derived layer targets are used for pretraining before
  mixed real/synthetic training.

### LSN

- The model extends the decomposition to lower bone, upper bone, and soft
  tissue layers.
- The published reconstruction is `R = 1 - product_i(1 - L_i)`.
- Paper training uses random bone-layer shifting and a two-stage pseudo-image
  curriculum.
- The current public repository's `random_movement` affine block is commented
  out; the active function only multiplies by the context mask. Directly
  executing the repository therefore does not reproduce actual random shifts.
- Both repositories require bone masks as model inputs. They cannot be copied
  as GT-free RAM inference modules without replacing GT masks by predicted
  per-bone probabilities.

## R326 adaptation

- Frozen mature R285 nnU-Net supplies fourteen base logits.
- A compact post-logit adapter remains active at inference and consumes only
  the image, base probabilities, and an expected-overlap map.
- It predicts bounded logit residuals and fourteen radiographic bone layers.
- Training-only pseudo layers split optical density equally among every GT
  bone membership at a pixel.
- A transmission reconstruction, layer L1, and image-gradient consistency form
  the projection prior. No validation/test GT is used by inference.
- A capacity-matched segmentation-only adapter is trained from the same
  initialization to isolate gains caused by extra parameters.

## Gradient calibration

- Native adapter gradient norm: `0.000712566`.
- Projection gradient norm: `0.900192`.
- Uncalibrated weight `0.05` would produce a prior/native ratio of `63.17`.
- Fixed training-only calibrated weight: `0.00025`, initial ratio `0.3158`.

## Validation results

| Configuration | Macro DSC | Macro IoU | Overlap DSC | Overlap IoU | Overlap NSD@2px | Overlap MSD px (lower better) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen mature baseline | 0.978139281 | 0.957428992 | 0.872652063 | 0.774075424 | 0.878223678 | 1.131774184 |
| Segmentation-only adapter | 0.978159189 | 0.957466602 | **0.873004960** | **0.774630941** | **0.879019930** | 1.125427198 |
| Layer-projection adapter | **0.978162825** | **0.957473636** | 0.872921727 | 0.774499889 | 0.878769003 | **1.124241192** |

### Layer projection minus frozen baseline

- Macro DSC: `+0.000023544`.
- Macro IoU: `+0.000044644`.
- Overlap DSC: `+0.000269665`.
- Overlap IoU: `+0.000424465`.
- Overlap NSD@2px: `+0.000545325`.
- Overlap MSD: `-0.007532992 px`.

### Layer projection minus capacity-matched adapter

- Macro DSC: `+0.000003636`.
- Macro IoU: `+0.000007033`.
- Overlap DSC: `-0.000083232`.
- Overlap IoU: `-0.000131052`.
- Overlap NSD@2px: `-0.000250927`.
- Overlap MSD: `-0.001186006 px`.

## Decision

R326 is a **mechanism-positive but claim-no-go** result.

The complete model beats the frozen mature baseline on every listed metric and
satisfies the hard Macro DSC/IoU non-degradation requirement. However, the
capacity-matched adapter is better on Overlap DSC/IoU/NSD. Consequently, the
experiment verifies that a small inference-time adapter is useful and that the
layer prior slightly improves global area metrics and MSD, but it does not yet
prove that BLS/LSN-style reconstruction is responsible for better overlap-mask
recovery.

The likely bottleneck is the equal optical-density pseudo target: it is a
stable mathematical decomposition but not identifiable from a single
radiograph and may assign physically incorrect shares to two overlapping
bones. A stronger follow-up needs trustworthy separated-layer supervision
(LSN synthetic pairs or CT/DRR) or an overlap-only reliability gate, rather
than merely increasing the projection coefficient.

## Artifacts

- Result: `outputs/ram_w600/r326_layer_projection_prior_batched/result.json`.
- Adapter checkpoints and histories are in the same output directory.
- Full remote log synchronized to
  `outputs/bridge_logs/r326_layer_projection_prior_batched.log`.
