# R333 Dual-Interaction Prior Experiment Plan

**Problem**: A background seam and a true projection overlap require opposite local decisions.
**Method thesis**: A shared local interaction refiner can combine an input seam probability map and an output completion prior when a label-derived state mask makes their losses mutually exclusive.
**Date**: 2026-08-12

## Claim map

| Claim | Minimum evidence |
|---|---|
| C1: seam and overlap priors can coexist without gradient conflict | Removing state exclusion must be worse than the gated combined loss |
| C2: the framework transfers across label representations | TSRS R201 validation and RAM official validation both pass Dice/IoU non-degradation gates and improve their decisive local metrics |

## Method

The common objective is

`L = Lseg + lambda_sep Lsep(Ssep) + lambda_comp Lcomp(Soverlap) + lambda_cons Lnoninterference`.

- `Ssep`: high seam-prior confidence restricted to label-confirmed background.
- `Soverlap`: pixels with at least two positive instance channels; available only when the label representation supports it.
- `Suncertain`: all remaining interaction pixels; no strong directional loss.
- `Ssep * Soverlap = 0` is asserted for every training sample.

RAM uses its 14-channel multi-label masks for both states. TSRS uses reliable binary foreground/background for separation and synthetic foreground corruption for completion; its unreliable color identity is not treated as true overlap supervision.

## Must-run order

1. CUDA smoke on both dataset adapters and assert finite losses, nonzero refiner gradients, and zero state-mask conflict.
2. RAM validation: R332 initialization, combined gated loss, fixed full budget, test locked.
3. TSRS original-val: R317 initialization, combined gated loss, full R201 metrics, clean-test-v2 locked.
4. Only if both validation gates pass: locked final evaluation on RAM test and TSRS clean-test-v2.

## Gates

### RAM

- Overall DSC and IoU must not fall below R325.
- Prior-region FP must decrease relative to R332 without losing its overlap DSC/NSD/MSD gains.
- Report official overall, overlap-region, and pair-intersection DSC/NSD/VOE/MSD/RAVD.

### TSRS

- Dice and IoU must exceed the matched R317 baseline.
- Primary gain: HD95/ASSD and Surface Dice.
- Diagnostics: gap-region FP, component merge rate, component-count MAE.
- Original-val only during development; clean-test-v2 remains untouched until lock.

## Anti-claims

- Do not claim one unconditional loss solves both states.
- Do not interpret TSRS color collisions as true projection overlap.
- Do not tune dataset-specific pixel radii without scale normalization.
- Do not use either test split for weight, epoch, threshold, or correction-depth selection.
