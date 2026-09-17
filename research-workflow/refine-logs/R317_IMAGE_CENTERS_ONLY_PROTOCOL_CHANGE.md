# R317 Image-Centers-Only Protocol Change

Date: 2026-07-29

## Decision

Remove age and sex conditioning from R317. Earlier project experiments did not
show reliable added value from demographic conditioning, and those variables
are unavailable in the intended cross-dataset setting.

The recovered R258B result formally recorded
`no_go_relation_prior_not_validated`: age/sex changed AUROC by only `+0.00323`
and close-pair recall by `+0.01823`, both below the preregistered added-value
gate, while negative specificity decreased by `-0.02850`, exceeding the allowed
non-inferiority margin. R259/R260 nevertheless generated their frozen maps from
the conditioned branch. R317 must not inherit that historical branch.

The final prior is now based only on:

- the local high-resolution X-ray crop;
- the low-resolution global-context crop;
- the two prediction-derived center maps.

No age or sex value is supplied during prior training or full-image prior-map
generation. The metadata feature vector is fixed to zero.

## Preserved protocol

- Dataset: `TSRS_RSNA-Epiphysis` only, 875 train / 96 original-val.
- No `TSRS_RSNA-Articular-Surface` files.
- No clean-test-v2 access.
- Three prior seeds: 2581, 2582, and 2583.
- Eighteen epochs with per-epoch prior checkpoints.
- Safety-gated prior checkpoint selection.
- Frozen three-seed prior-map ensemble.
- Mature R202 nnU-Net initialization.
- Continuous prior loss with `alpha=0.035` as the controlled first test.
- Periodic nnU-Net checkpoints and complete R201 original-val selection.

## Transition integrity

The original R317 process first trained the image-centers control branch. The
completed seed 2581 run is reused exactly. The switch waits for seed 2582 to
complete and write its selected checkpoint, then terminates the old process
before demographic training. Seed 2583 is trained by the new image-centers-only
launcher.

Incomplete seeds are never resumed with a reset AdamW optimizer. They are
deleted and retrained from epoch 1. This avoids silently changing the optimizer
trajectory.

## Isolated outputs

- Prior models: `outputs/pair_prior/r317_image_centers_only`
- Prior maps: `outputs/priors/r317_image_centers_only`
- nnU-Net: `outputs/nnunet/r317_image_centers_only`
- Log: `outputs/bridge_logs/r317_image_centers_only.log`
- Result: `outputs/analysis/r317_image_centers_only_checkpoint_selection.json`
