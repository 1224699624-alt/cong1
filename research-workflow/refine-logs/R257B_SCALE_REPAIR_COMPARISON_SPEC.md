# R257b Frozen-Center Scale Repair Comparison

## Scope

- Reuse the fixed R257 epoch-8 image-only center detector and its fixed center threshold/NMS/mask gate.
- Dataset: `TSRS_RSNA-Epiphysis_contrast_v1` train/original-val only.
- No clean-test-v2, original test, Articular-Surface, age/sex input, or GT-derived inference crop.
- The same predicted center coordinates and the same GT one-to-one matching are used for both scale methods.

## Compared scale methods

### A. Instance-balanced log-scale head

- Freeze the complete R257 foreground/center backbone.
- Replace only the scale head and train it for four fixed epochs.
- Target: standardized `log(sqrt(instance area in letterbox pixels))`.
- Loss: instance-equal Smooth L1 sampled only at each GT instance center pixel.
- No sigmoid scale parameterization and no validation checkpoint selection.

### B. Prediction-derived Voronoi/basin scale

- No additional learning.
- Threshold the frozen R257 foreground probability at fixed `0.50`.
- Assign every predicted foreground pixel to its nearest predicted center in letterbox coordinates.
- Proposal scale is the square root of its assigned foreground basin area.

## Paired audit

- Both methods use identical R257 centers; only their scale values differ.
- Report global, relative-close endpoint, and contact endpoint absolute log scale errors.
- Re-report the shared center recall/precision, close/contact pair coverage, count MAE, false proposals/image, and localization error to verify center invariance.
- Include R257's original sigmoid-scale result as a frozen reference, not as a selectable checkpoint.

## Gate

A method passes scale repair only if:

- Global median absolute log scale error <= `0.50`.
- Close-endpoint median absolute log scale error <= `0.60`.
- Contact-endpoint median absolute log scale error <= `0.60`.
- Shared center recall >= `0.85`, precision >= `0.75`, close coverage >= `0.80`, and contact coverage >= `0.75`.
- All 96 original-val images are evaluated and all required metrics are finite.

Any limited run is forced to `sanity_only`. Passing permits only exploratory R258 OOF proposal generation with the passing scale method; it does not permit nnU-Net integration or clean-test-v2 evaluation.
