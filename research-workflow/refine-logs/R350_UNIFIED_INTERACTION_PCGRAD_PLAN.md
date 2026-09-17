# R350 Unified Interaction PCGrad Plan

**Problem:** R349 ran both datasets, but TSRS had no overlap/completion state and therefore did not test the same three-state loss as RAM.

**Primary claim:** One state-conditioned loss construction can preserve mature segmentation while emphasizing separation or overlap according to the reliable local states present in each dataset.

**Anti-claim:** R350 does not claim that TSRS supplies RAM-style multi-label projection-overlap ground truth. TSRS uses a conservative foreground-completion proxy inside a seam-prior corridor; RAM uses its independent instance outputs.

## Shared construction

`L = Lbase + wsep Lsep + wov Lov + wpreserve Lpreserve`

- separation and overlap states are mutually exclusive;
- all state losses are normalized by active mass;
- the effective structural weight is `lambda * min(1, active_mass / 0.01)`;
- the formula and base structural coefficient (`0.035`) are identical;
- parameter-space PCGrad removes gradients opposing the base segmentation loss;
- each auxiliary gradient is capped at 15% of the base-gradient norm.

The mass rule produces the dataset emphasis without a dataset-name branch. Expected behavior is separation-dominant TSRS and overlap-dominant RAM.

## State adapters

### TSRS

- Separation: high-confidence seam corridor intersecting GT background.
- Overlap/completion proxy: seam corridor intersecting GT foreground and frozen-R317 foreground confidence >= 0.75.
- Preserve: all pixels outside those two states, using full two-class distillation.
- Initialization/baseline: the same mature R317 checkpoint used by R349.

### RAM

- Separation: seam corridor intersecting background and excluding predicted overlap.
- Overlap: frozen-R332 second-instance probability >= 0.25.
- Preserve: all pixels outside those states, using per-instance distillation.
- Initialization/baseline: frozen R332 two-step checkpoint.

## Evaluation

- TSRS: original-val only, best and final checkpoints, full R201 metrics. `clean-test-v2` remains locked.
- RAM: validation only, official overall/overlap/pair metrics plus seam FP. RAM test remains locked.
- Dataset structures are unchanged; all outputs are isolated under R350 folders.

## Gates

- Initial prediction identity must be exact.
- Both state types must have nonzero coverage in the real-data audit.
- Gate overlap must be zero.
- Dice and IoU must not fall below the matched mature baseline.
- TSRS should improve at least one separation/boundary metric without damaging the other decisive metrics materially.
- RAM overlap DSC/NSD/MSD and seam FP must all be non-worse than R332.

## Run order

1. CPU formula smoke.
2. RAM and TSRS real-data CUDA audits.
3. TSRS 30 epochs, then best/final original-val R201 evaluation.
4. RAM 30 epochs and validation model selection.
