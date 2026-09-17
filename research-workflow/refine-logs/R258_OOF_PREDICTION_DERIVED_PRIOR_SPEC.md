# R258 Patient-Disjoint OOF Prediction-Derived Prior

## Phase A: OOF image-only proposals

- Scope: `TSRS_RSNA-Epiphysis_contrast_v1` train only for OOF generation; original-val remains the downstream relation-prior audit split.
- Build five deterministic patient-disjoint folds, stratified by sex and train-only bone-age bins.
- For each fold, train a fresh R257 center detector from scratch on the other four folds for eight fixed epochs.
- Generate held-out centers, foreground, and R257b primary basin scales only with the fold model that never saw that image.
- Every one of the 875 train images must appear in exactly one held-out fold; no full-train checkpoint may generate train proposals.
- Fixed R257 thresholds/NMS/mask gate and fixed basin threshold `0.50`; no fold-specific tuning or checkpoint selection.

### OOF gate

- 875/875 unique train images covered once.
- Center recall >= `0.85`, precision >= `0.75`.
- Close-pair coverage >= `0.80`, contact-pair coverage >= `0.75`.
- Global/close/contact median absolute log basin-scale errors <= `0.50/0.60/0.60`.
- Count MAE <= `3.0`, false proposals/image <= `3.0`.
- Fold checkpoints, fold manifest, OOF proposal CSV, hashes, and aggregate JSON are isolated and machine-readable.

## Phase B: simplified development-conditioned relation prior

Phase B is blocked until Phase A passes. It will use:

- OOF train proposals from Phase A.
- Frozen full-train R257 centers plus R257b basin scales on original-val.
- Local X-ray plus two predicted-center maps as model input.
- Matched comparison: image+centers versus image+centers+continuous bone age+sex.
- No explicit geometry vector, because R256 showed it was redundant.
- GT only labels proposal provenance and train/val audit targets after prediction-derived candidates exist.

No clean-test-v2, original test, Articular-Surface, nnU-Net integration, or segmentation claim is allowed in R258.
