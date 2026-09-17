# R351 evaluation and result-integrity notes

## Scope

R351 adds evaluation-only guards and reusable functions. It does not train a
model, open RAM test data, modify the raw dataset, or rewrite any previous
result. The implementation is in
[`scripts/r351_evaluation.py`](../../scripts/r351_evaluation.py), with a
dependency-light test file at
[`scripts/test_r351_evaluation.py`](../../scripts/test_r351_evaluation.py).

## Fixed RAM seam/background ROI

For each case, the fixed ROI is

```text
minmax(seam_prior) >= seam_threshold AND NOT(GT_union)
```

where `GT_union` is the logical union over the available RAM instance mask
channels. The default seam threshold is `0.25`, matching the R350
per-image seam normalisation convention. The ROI is materialised before any
prediction is inspected. A prediction can change FP counts, but it cannot
change the ROI. Every row reports:

- `fp_pixels`: predicted union pixels inside the ROI;
- `roi_area`: fixed ROI pixel count;
- `fp_rate`: `fp_pixels / roi_area` for a non-empty ROI;
- `fp_over_area`: an explicit `"FP/area"` string;
- `roi_checksum`: a SHA-256 checksum of the fixed boolean ROI.

Empty ROI handling is explicit. A per-image empty ROI has `fp_pixels = 0`,
`roi_area = 0`, and `fp_rate = null`; it is not treated as a zero-error
improvement. Aggregate `fp_rate` is pooled FP pixels divided by pooled area
over non-empty ROIs. The aggregate also reports the number of empty and
non-empty images, and reports the unweighted mean of non-empty per-image
rates separately.

The main interfaces are:

```python
fixed_seam_background_roi(seam_prior, gt_union, seam_threshold=0.25)
evaluate_fixed_seam_roi(predictions, seam_priors, targets, ...)
compare_fixed_seam_roi(baseline_predictions, adapted_predictions,
                       seam_priors, targets, ...)
```

The three collection mappings must have identical case IDs. Both arms in a
comparison are evaluated against the same prior/GT-derived checksums; a
checksum mismatch raises instead of silently comparing different ROIs.

## Acceptance semantics

`check_dice_iou_non_degradation` is strict by default:

```text
formal pass: candidate Dice - baseline Dice >= 0
             candidate IoU  - baseline IoU  >= 0
formal_tolerance = 0.0
```

The historical `0.0005` value is retained only as the default
`near_tie_tolerance`. A negative delta inside that band is labelled
`near_tie` and still fails the formal pass. A negative delta is never labelled
as an improvement. A caller may explicitly pass a non-zero formal tolerance,
but that changes the declared gate and must be recorded by the caller.

`assess_comparison` reports all scalar metrics it receives, including their
candidate-minus-baseline delta, direction, status, and favorable flag. Its
domain-specific primary metrics are:

- TSRS: lower `gap_region_fp_rate`/`gap_fp` and lower
  `component_merge_rate`/`merge_rate`.
- RAM: higher overlap DSC (and optional IoU/NSD when present), lower overlap
  MSD, and lower pair-intersection MSD.

The formal claim gate requires strict Dice/IoU safety plus every required
primary metric to be favorable or unchanged, with at least one primary metric
strictly improved. Optional metrics are still reported. Auxiliary metrics are
listed separately, including `auxiliary_degraded`. The result explicitly sets
`universal_metric_non_degradation = false`; a positive primary result does not
promise that every boundary, surface, or other auxiliary metric improves.
No statistical-significance claim is made by this module.

## Single-epoch result binding

`bind_single_checkpoint_result` and `write_bound_result` create and validate a
`R351_RESULT_V1` envelope with:

```text
selection.epoch
selection.checkpoint_sha256
metrics
```

The SHA must be a 64-character hexadecimal SHA-256; `sha256_file` computes it
for a checkpoint. Embedded `epoch`/`epochs` entries in the metrics object must
match the selected epoch, so an accidentally supplied history containing
multiple epochs is rejected. The resulting JSON therefore binds one metrics
object to one selected epoch and one checkpoint hash. The R351 training
scripts' existing `best.pth`/history output was not edited; callers should
wrap the selected row with this envelope when producing a paper-facing result.

## Read-only overlap-pretraining audit

The requested source review covered `scripts/r351_data.py`,
`scripts/train_r351_overlap_prior.py`, `scripts/r351_dual_prior.py`, the
`NativeWristDataset` parent, and the R322/R323 map builders.

- The overlap pretraining target is true RAM multi-label supervision:
  `mask.sum(1, keepdim=True) >= 2`. No teacher prediction is used as the
  target. The script constructs separate `train` and `val` datasets and never
  loads `test`.
- Validation runs with `model.train(False)` and
  `torch.set_grad_enabled(False)`, so it does not update weights. The valid
  padded-image mask is downsampled with nearest interpolation and masks the
  BCE reduction.
- `AlignedRamDataset` deliberately rejects `augment=True`. This is correct
  for a frozen spatial prior: the image, GT mask, and prior receive no
  unsynchronised geometric transform.
- `NativeWristDataset` horizontally flips `_R` image/mask pairs. The R322/R323
  prior builders canonicalise `_R` images before generating their maps, so the
  adapter's decision not to flip the loaded prior a second time is aligned for
  those maps. A prior made by a different orientation pipeline must not be
  mixed in without a spatial audit.
- A preflight gap remains in the existing dataset wrapper: it does not check
  train/val prior coverage during `__init__`; a missing `.npy` and `.png` is
  only discovered when that case is fetched. This is a launch blocker for any
  prior root without complete coverage. In the current local workspace,
  `outputs/priors/r323_ram_native_r317_single_seed` contains only its manifest
  (no train/val maps), whereas `outputs/priors/r322_ram_r317_seam` contains
  425 train and 69 val maps. Verify the selected prior root before launching a
  short run.

## Verification

```powershell
python -m py_compile scripts/r351_evaluation.py scripts/test_r351_evaluation.py
python scripts/r351_evaluation.py --self-test
python scripts/test_r351_evaluation.py
python -m unittest discover -s scripts -p 'test_r351_evaluation.py'
```

The dependency-light suite covers fixed-ROI invariance, per-image FP/area,
pooled aggregation, empty-ROI semantics, strict-vs-near-tie overlap gates,
TSRS and RAM primary/auxiliary reporting, case-key checks, SHA binding, and
mixed-epoch rejection.
