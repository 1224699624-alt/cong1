# R285 RAM-W600 nnU-Net Overlap-Prior Plan

- Dataset: RAM-W600 official 425/69/124 split; original data remains read-only.
- Backbone: nnU-Net v2 `PlainConvUNet`, 2D, six stages, 32--320 channels,
  InstanceNorm and LeakyReLU.
- Required adaptation: 14 independent sigmoid bone outputs because RAM-W600 masks overlap;
  default mutually exclusive nnU-Net softmax is invalid for this label representation.
- Baseline and improved models share the same 15-output architecture. The baseline ignores
  the overlap channel. The improved model learns the 15th overlap-prior channel and injects it
  through a zero-initialized 1x1 residual into the 14 bone logits.
- Loss, train-only top-15 anatomical pairs, validation non-inferiority selection, fixed 0.5
  threshold, official test metrics and visualizations match R284b.
- Outputs are isolated under `outputs/ram_w600/r285_nnunet_overlap_prior`.
