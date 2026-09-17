# R284 RAM-W600 Multi-Label Overlap-Prior Pilot

- Dataset: `G:\gutou\RAM-W600`, read-only; official split 425 train / 69 val / 124 test.
- Baseline: one-channel X-ray to 14 independent sigmoid bone masks using a compact U-Net.
- Improved: initialize from the paired best baseline; add an image-conditioned overlap-prior
  head and zero-initialized residual segmentation branch.
- Loss: multi-label Dice+BCE plus fixed `0.10` overlap-prior supervision and fixed `0.05`
  pairwise overlap consistency for the 15 most frequent train-only anatomical pairs.
- Selection: test set is not used for checkpoint selection. The improved checkpoint first
  satisfies validation macro-Dice non-inferiority (`-0.002`), then maximizes overlap Dice;
  threshold is fixed at `0.5` and is not searched.
- Final metrics: 14-bone macro/per-class Dice, overall overlap Dice/NSD 2 px, and
  per-pair overlap Dice/NSD 2 px on the official test split.
- Outputs are isolated under `outputs/ram_w600/r284_overlap_prior`.
