# R257 Image-Only Epiphysis Center/Scale Detector

## Scope

- Dataset: `TSRS_RSNA-Epiphysis_contrast_v1` train/val only.
- Input: contrast-normalized X-ray only; no mask, age, sex, bone age, or GT-derived crop at inference.
- GT instance labels are used only to construct train targets and original-val matching/audit metrics.
- No Articular-Surface, original test, or clean-test-v2.
- Outputs and checkpoints are isolated under the R257 namespace.

## Question

Can an image-only detector recover enough real epiphysis center/scale proposals to replace R256's GT-jitter oracle proposals, especially for close/contact bone pairs?

## Fixed representation

- Preserve aspect ratio by deterministic square letterboxing; do not spatially align different hands to a common atlas.
- A small U-Net predicts three dense fields: bone foreground probability, instance-center probability, and local instance scale.
- Center targets are per-instance Gaussians; scale targets are normalized square-root instance area around each center.
- Fixed final epoch, seed, center threshold, NMS kernel, mask gate, and maximum proposal count. No original-val threshold search or checkpoint selection.

## Metrics

- One-to-one center recall/precision using Hungarian matching with a scale-normalized tolerance.
- Center localization error normalized by GT instance scale.
- Median absolute log scale error for matched centers.
- Close/contact endpoint localization and scale errors are reported separately so easy isolated bones cannot hide difficult-pair failures.
- Proposal-count MAE and false proposals per image.
- Relative-close pair endpoint coverage and contact-pair endpoint coverage: a GT pair is covered only when both endpoints match two distinct predicted centers.
- All metrics are computed against original-val GT only after image-only proposals have been generated.

## Gate

- Center recall >= `0.85` and precision >= `0.75`.
- Relative-close pair coverage >= `0.80`.
- Contact-pair coverage >= `0.75`, with a nonzero contact denominator.
- Mean normalized localization error <= `0.50`.
- Median absolute log scale error <= `0.50`.
- Close/contact endpoint mean normalized localization error <= `0.60/0.65` and median absolute log scale error <= `0.60/0.60`.
- Proposal-count MAE <= `3.0` and false proposals/image <= `3.0`.
- All metrics finite; all 96 original-val images evaluated; no NaN/OOM/error.

Passing permits only an exploratory R258 feasibility run: generate patient-disjoint OOF train proposals and original-val image-only proposals, then retrain the simplified R256 relation prior without the redundant explicit geometry vector. The `0.80/0.75` coverage gate does not establish that image-only proposals have replaced the R256 oracle proposals. It does not permit nnU-Net integration or clean-test-v2 use. Any limited sanity run is forced to `decision=sanity_only` and can never pass the formal gate.
