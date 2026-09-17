# R256 Scale-Invariant Development-Conditioned Pair Prior Pilot

## Scope

- Dataset: `TSRS_RSNA-Epiphysis_contrast_v1` train/val only.
- Labels: original instance-valued train/val labels copied unchanged.
- Metadata: `filtered_train.csv`, `filtered_val.csv`.
- No original test, no clean-test-v2, no Articular-Surface.

## Question

As an oracle-metadata mechanism pilot, can a local bone-pair model tolerate detector-like center/scale perturbations while learning (1) whether two candidate centers belong to distinct bone instances and (2) a broad separation probability field, and does continuous bone age plus sex improve this over image/geometry alone?

## Representation

- Local square crop around a candidate pair, resized to `128x128`.
- Input channels: enhanced X-ray, Gaussian center A, Gaussian center B.
- Relative features: center distance normalized by local bone scale, log area ratio, relative displacement.
- Development features: continuous bone age and sex.
- Positive pairs: nearest distinct GT instances, converted into noisy center/scale proposals.
- Negative pairs: two pseudo-centers sampled from opposite halves of one GT instance and passed through the same proposal perturbation.
- Heatmap target: soft, broad equidistant background corridor for positive pairs; zero for same-instance negatives.
- This pilot deliberately seeds candidate centers/scales from GT instances and then applies fixed detector-like perturbations. Thus it is an oracle-proposal mechanism study, not a deployable image-only method. GT also supplies pair labels and heatmap targets. Crop and model features read only the perturbed proposal, never the unperturbed masks/bounding boxes.

## Matched comparison

1. `geometry_only`: image plus relative geometry; age/sex zeroed during both train and val.
2. `development_conditioned`: identical architecture/training plus normalized bone age and sex.

No validation threshold search and no model selection: fixed epochs, fixed threshold `0.5`, proposal seed `256`, and training seeds `256/257/258`. Bone age/sex are oracle teacher metadata in this pilot and are not a deployable inference input.

## Gate

- Full original-val pair AUROC and AP are finite.
- Scale-normalized close distinct-pair recall, contact-pair recall, and same-instance specificity are all reported; close recall and specificity are at least `0.75`.
- Positive-pair continuous heatmap Dice is at least `0.20`.
- Nonzero heatmap-target coverage is at least `0.85`; zero-target/contact pairs are never silently excluded.
- Proposal coverage is reported against all scale-eligible GT pairs before capping; contact coverage is `1.0` and relative-close coverage is at least `0.95`.
- Development-conditioned AUROC improves by at least `0.005` or close-pair recall by at least `0.02`, without reducing specificity by more than `0.02`.
- Positives and same-instance negatives are balanced within each included case.
- Same-instance pseudo-negative proposals are greedily matched to positive proposals by normalized center distance and absolute log scale ratio; the validation standardized mean difference must not exceed `0.50`.
- Performance is reported under fixed center/scale perturbation, three training seeds, case-level paired bootstrap, and X-ray/center/geometry/metadata-only/shuffled-metadata shortcut ablations.

Passing this gate permits only implementation of an image-only center/scale detector. It does not permit nnU-Net loss integration or support a segmentation-improvement claim.
