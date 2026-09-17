# R287 Pair-Specific Boundary Prior Development Plan

- Dataset: RAM-W600 official train/validation only during development.
- Baseline: mature R285 nnU-Net checkpoint.
- Backbone: fully frozen for all epochs; no decoder fine-tuning.
- Prior: 15 pair-specific `gap/overlap/uncertain` local-interface maps.
- New loss: differentiable pair-overlap morphological-boundary Dice plus balanced BCE,
  weight `0.01` fixed before launch.
- Selection: validation macro Dice must remain within `0.0002` of baseline; among eligible
  checkpoints maximize validation overlap NSD 2 px, then overlap Dice.
- Fixed threshold: `0.5`; no threshold search.
- Official test: locked and not evaluated in R287 development.
- Outputs: isolated under `outputs/ram_w600/r287_pair_boundary_prior_dev`.

## Running status

- Local CUDA process started successfully; PID `9656`.
- Epoch 0 passed the macro gate.
- Validation macro Dice: `0.978138208 -> 0.978129983` (`-0.000008225`).
- Validation overlap Dice: `0.872642507 -> 0.872664172` (`+0.000021665`).
- Validation overlap NSD 2 px: `0.878208674 -> 0.878348088` (`+0.000139415`).
- This is an early development signal only; official test remains unevaluated.

## Final development result

- Status: complete without errors; official test remained locked.
- Selected checkpoint: relation warm-up epoch 2 by validation overlap NSD 2 px.
- Validation macro Dice: `0.978138208 -> 0.978133023` (`-0.000005186`).
- Validation overlap Dice: `0.872642507 -> 0.872852205` (`+0.000209698`).
- Validation overlap NSD 2 px: `0.878208674 -> 0.878530276` (`+0.000321603`).
- Later epochs continued to increase overlap area Dice but reduced boundary NSD, confirming
  that early stopping on the boundary metric is necessary.
- Strong positive pair-level NSD signals include Capitate--Hamate (`+0.001234`),
  Lunate--Scaphoid (`+0.001225`), and Trapezium--Metacarpal1 (`+0.000981`).
- Negative pair-level outliers remain, especially Capitate--Scaphoid (`-0.001030`).

Decision: the pair-boundary loss fixes R286b's validation surface-metric failure while
preserving macro Dice to within `6e-6`. Keep the official test locked until the remaining
pair-specific negative outliers are addressed or the configuration is otherwise frozen.
