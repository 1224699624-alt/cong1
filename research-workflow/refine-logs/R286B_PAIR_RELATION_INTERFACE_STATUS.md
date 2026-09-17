# R286b Pair-Relation Local-Interface Result

- Status: complete locally on CUDA; no runtime error.
- Process metadata: `outputs/ram_w600/r286b_pair_relation_interface_prior/local_process.json`.
- Baseline: mature R285 nnU-Net checkpoint; initial improved logits are exactly identical.
- Protocol: 15 training-selected bone pairs, three relations per pair, radius-7 local ROI,
  fixed threshold 0.5, validation-only checkpoint selection, test untouched until selection.
- Training: 8 relation-head warm-up epochs, then up to 20 low-LR decoder fine-tuning epochs.

## First warm-up epoch

- Validation macro Dice: `0.978135586` versus baseline `0.978138208` (`-0.000002623`).
- Validation overlap Dice: `0.872711893` versus baseline `0.872642507` (`+0.000069387`).
- Macro eligibility gate: passed.
- Mean learned overlap gain: `0.0426603`; mean gap gain remains `0` on RAM-W600.

R286b corrected the original R286 checkpoint-ordering bug so that macro Dice is an
eligibility gate and overlap Dice is the optimization target among eligible checkpoints.
The one-epoch R286 log is retained for audit, but R286b is the valid run.

## Final checkpoint and official test

- Selected checkpoint: relation-head warm-up epoch 6; no decoder-finetune checkpoint passed
  the macro-Dice gate.
- Validation macro Dice: `0.978138208 -> 0.978129745` (`-0.000008464`).
- Validation overlap Dice: `0.872642507 -> 0.872953047` (`+0.000310540`).
- Test macro Dice: `0.978429914 -> 0.978422940` (`-0.000006974`).
- Test overlap Dice: `0.868747911 -> 0.869124167` (`+0.000376256`).
- Test overlap NSD 2 px: `0.877936200 -> 0.877749215` (`-0.000186985`).

## Interpretation

Pair-specific relation learning gives a small positive overlap-Dice signal while preserving
macro Dice to within `7e-6`. However, overlap surface quality does not improve, and unfreezing
the final decoder stages immediately degrades both macro and overlap metrics. The useful
component is therefore the frozen-backbone relation residual, not decoder adaptation. This
run is promising mechanistic evidence but not yet a paper-level improvement because the gain
is small and NSD is negative.
