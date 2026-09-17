# R257 Image-Only Center/Scale Detector Result

## Status

- Completed: 2026-07-14 20:29
- Scope: `TSRS_RSNA-Epiphysis_contrast_v1` train/original-val only
- Input at inference: X-ray only
- Fixed epoch: 8; fixed threshold/NMS; no validation selection
- clean-test-v2 used: no
- Articular-Surface used: no
- Official decision: `no_go_image_only_proposals_insufficient`

## Final original-val metrics

| Metric | Result | Gate | Status |
|---|---:|---:|---|
| Center recall | 0.983693 | >= 0.85 | pass |
| Center precision | 0.921696 | >= 0.75 | pass |
| Normalized localization error | 0.114630 | <= 0.50 | pass |
| Close-pair coverage | 0.953725 | >= 0.80 | pass |
| Contact-pair coverage | 0.906250 | >= 0.75 | pass |
| Proposal-count MAE | 1.968750 | <= 3.0 | pass |
| False proposals/image | 2.135417 | <= 3.0 | pass |
| Global median abs log scale error | 1.451902 | <= 0.50 | fail |
| Close-endpoint median abs log scale error | 2.161196 | <= 0.60 | fail |
| Contact-endpoint median abs log scale error | 2.220379 | <= 0.60 | fail |

The image-only center detector is strong: it matched 2,413/2,453 GT centers and covered 845/886 close pairs and 232/256 contact pairs. The scale head is not usable. An absolute log error of `2.22` corresponds to approximately a `9.2x` multiplicative scale error on contact endpoints.

## Training behavior

Scale error temporarily reached `0.478079` at epoch 2 but deteriorated afterward, while center recall/precision and count metrics remained strong. The protocol fixes epoch 8 and forbids selecting epoch 2 post hoc. This pattern indicates an unstable/mis-specified scale-regression head rather than failure of image-only center localization.

## Decision

Do not start R258 with the current R257 scale values and do not use clean-test-v2. Preserve the final center checkpoint/proposals as evidence that GT-free center generation is viable. The next isolated experiment should repair scale estimation, preferably by log-scale regression at center locations or by deriving scale from a predicted foreground basin, while freezing the already successful center proposal rules. Only after scale metrics pass may R258 generate patient-disjoint OOF train proposals.
