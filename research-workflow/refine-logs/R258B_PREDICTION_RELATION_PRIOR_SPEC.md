# R258B Prediction-Derived Relation Prior Preregistration

## Scope and provenance

- Dataset: `TSRS_RSNA-Epiphysis_contrast_v1` only.
- Train inputs: fixed R258 5-fold OOF proposal CSV over 875 train images.
- Validation inputs: fixed frozen-R257/R257b basin proposal CSV over 96 original-val images.
- GT is accessed only after proposal generation to assign pair supervision and separation-heat targets.
- Pinned train/validation proposal SHA256 values are `23ba7f570e86b6a5807c50ca5097e0a8b0040fdf7f72f0748a794ec0f98c056e` and `5f17a815078ab7d275f01565cb82f84f6bd346b09b3d8e8e22e1f140c008b634`. Full execution must also reproduce the Phase A fold/inference manifests and authorization decision.
- `clean-test-v2`, original test, and Articular-Surface are prohibited.

## Matched comparison

1. `image_centers`: local X-ray crop plus two prediction-derived center maps.
2. `image_centers_age_sex`: identical inputs/model/training plus continuous bone age and binary sex.

No center distance, direction, scale ratio, or other explicit geometry vector is provided to either model. Prediction-derived basin scale is used only to define a scale-normalized crop and center-map placement.

## Pair labels

- Proposals are assigned to GT instances only for supervision, with fixed normalized tolerance `0.60`.
- Positive: two local proposals correspond to different GT bones whose relative boundary gap is at most `0.80`.
- Negative: local proposal pair is not a valid positive pair, including duplicate/unmatched proposals and distinct bones outside the supervised local relation.
- Positives and geometry-matched negatives are balanced per image, capped at 32 of each.
- No GT coordinate or mask is a model input.

## Fixed training protocol

- Crop `128 x 128`; batch size `32`.
- Four fixed epochs; no checkpoint or epoch selection.
- Seeds `2581,2582,2583`.
- AdamW, learning rate `5e-4`, weight decay `1e-4`.
- Classification BCE plus `0.35` separation-heat loss.
- Fixed decision threshold `0.50`; no threshold search.

## Gate

- All reported metrics finite.
- Development-conditioned close-pair recall >= `0.75`.
- Contact-pair recall >= `0.70`.
- Negative specificity >= `0.75`.
- Positive heatmap soft Dice >= `0.20`.
- Valid positive heat-target coverage >= `0.85`; invalid positive targets count as Dice zero in the reported heat Dice.
- Development conditioning adds AUROC >= `0.005` or close recall >= `0.02` versus the matched baseline.
- Specificity is no worse than baseline by more than `0.02`.
- Metadata-only AUROC <= `0.60`.
- Real-metadata AUROC exceeds shuffled-metadata AUROC by >= `0.003`.
- Three seeds complete.

Passing permits only a controlled prior/loss integration experiment on original train/val. It is not itself a segmentation result and does not unlock clean-test-v2.
