# R185/R186 Fast Layout Gate Results

Both runs were executed on the remote server only.

## R185 Component-Accepted Layout Cut

R185 attempted to turn the older distance-transform layout bridge cut into a stricter gate:

- validation must have non-zero edits;
- per-image edits must not hurt Dice beyond tolerance;
- per-image edits must not hurt Boundary IoU;
- per-image edits must not increase component-count error.

The route was stopped during smoke because even a small grid on full-resolution masks was too CPU-bound and produced no intermediate result. This is not a metric failure, but it is a workflow failure: it is not suitable for fast ARIS iteration without caching or downsampled precomputation.

Decision: do not run R185 full.

## R186 Fast Neck Gate

R186 replaced distance-transform cuts with a faster binary-opening neck candidate rule and added parameter-level progress output.

Smoke completed on the server. It found a non-zero validation edit:

- selected rule: `neck_width=2`, `band_radius=2`, `min_area=4`, `max_remove_frac=0.0005`
- validation edited fraction: `2.025825e-07`
- validation Dice delta vs sampled R110 anchor: `+0.0000046`
- validation Boundary IoU delta vs sampled R110 anchor: `+0.0000144`

Locked clean-test-v2 application:

- Dice: `0.9177202069995851`
- Dice delta vs R110: `-0.000002949353394066101`
- Boundary IoU: `0.25189431179029953`
- Boundary IoU delta vs R110: `-0.0000016986505581018108`
- component-count error: `2.037037037037037`
- false-bridge mean: `0.7037037037037037`
- edited fraction: `7.656069e-08`
- status: `below_best`

## Decision

Do not launch R186 full. The fast neck gate can slightly improve structural metrics, but the edit magnitude is too tiny and clean-test-v2 Dice does not improve. This is useful as evidence that simple geometry-only postprocessing cannot close the target gap.

The next route should move away from postprocessing sweeps. The strongest remaining options are:

1. a materially new model-family/environment route, preferably a faithful MaskDINO/Mask2Former-style setup with a CUDA-toolkit-capable isolated environment, or
2. a data/protocol route only after R168 reviewed CSV exists and R170 can build the reviewed train/val variant.
