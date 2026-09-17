# R285 RAM-W600 nnU-Net Overlap-Prior Result

- Completed locally with the official 425/69/124 split, fixed threshold `0.5`, and no
  test-set model selection or threshold search.
- Backbone: nnU-Net v2 2D `PlainConvUNet`, six stages, 32--320 channels, 14 sigmoid
  instance outputs plus one learned overlap-prior output.

## Official test result

- Macro Dice: `0.978430 -> 0.978175` (`-0.000255`).
- Any-overlap Dice: `0.868748 -> 0.868285` (`-0.000463`).
- Any-overlap NSD 2 px: `0.877936 -> 0.877168` (`-0.000768`).
- Validation also did not beat the paired baseline: macro Dice `0.978138 -> 0.977883`;
  overlap Dice `0.872643 -> 0.872567`.

## Comparison with R284b

- Compact U-Net baseline test: macro Dice `0.973944`, overlap Dice `0.830912`, overlap
  NSD `0.813016`.
- nnU-Net baseline test: macro Dice `0.978430`, overlap Dice `0.868748`, overlap NSD
  `0.877936`.
- The nnU-Net baseline already resolves much of the failure bracket that R284b's explicit
  overlap supervision corrected in the compact model.

## Decision

No-go for the current single overlap-map, scalar 1x1 residual injection on nnU-Net. The
negative deltas are small but fail the requirement that main Dice not decrease. This does
not refute relation-aware supervision: pair-level effects are mixed and the strong baseline
already predicts every selected overlap pair. A subsequent nnU-Net test should use
pair-specific relation features or decoder-level interface modulation rather than a single
global overlap channel.
