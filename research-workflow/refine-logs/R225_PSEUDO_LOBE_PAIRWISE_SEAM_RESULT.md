# R225 Pseudo-Lobe Pairwise Seam Result

Date: 2026-07-08

## Purpose

R224 showed a strong oracle upper bound when seam candidates are generated
between true GT-instance pairs. R225 tests whether a GT-free pseudo-instance
approximation can recover that benefit:

- split R110 anchor components into pseudo-lobes using distance-map peaks;
- generate seam candidates only between pseudo-lobe pairs;
- use GT only for original-val diagnostic labels and metrics.

This is original-val development evidence only. It does not use clean-test-v2
and writes no final masks.

## Implemented

Added:

- `scripts/audit_r225_pseudo_lobe_pairwise_seam_candidates.py`

The script:

- performs GT-free candidate generation from R110 anchor geometry;
- creates pseudo-lobes inside each large connected component;
- proposes pairwise seam cuts between pseudo-lobes;
- evaluates candidates with Dice, IoU, Recall, Boundary IoU/F1, gap FP, and
  component-count diagnostics;
- writes JSON/CSV only.

Artifacts:

- `outputs/analysis/r225_pseudo_lobe_pairwise_seam_val12_candidates.csv`
- `outputs/analysis/r225_pseudo_lobe_pairwise_seam_val12_summary.json`
- `outputs/logs/r225_pseudo_lobe_pairwise_seam_val12.log`

## Run Notes

A full-val screen was started but stopped early because the pseudo-lobe
candidate generator was too slow. Even the lightweight setting processed
roughly 12 images in about 8.5 minutes.

Completed diagnostic:

- split: original `TSRS_RSNA-Epiphysis/val`
- limit: `12`
- seam radius: `2`
- action fraction: `0.50`
- clean-test-v2 used: `false`
- writes masks: `false`

## Result

Coverage:

- candidate rows: `609`
- images: `12`

Overall candidates:

- quick-useful: `260/609`
- hard-risk: `340/609`
- safe-gap positive: `104/609`
- over-erosion proxy: `386/609`
- mean Dice delta: `-0.000070`
- mean IoU delta: `-0.000118`
- mean Recall delta: `-0.000216`
- mean Boundary IoU delta: `-0.000034`
- mean Boundary F1 delta: `-0.000032`
- mean gap-region FP delta: `-0.000194`
- mean component count MAE delta: `+0.011494`
- mean cut GT foreground fraction: `0.649982`
- mean cut GT gap fraction: `0.350018`

Safe-gap positive subset:

- rows: `104`
- images: `10`
- quick-useful: `90`
- hard-risk: `14`
- over-erosion proxy: `0`
- mean Boundary IoU delta: `+0.000523`
- mean Boundary F1 delta: `+0.000668`
- mean gap-region FP delta: `-0.000398`
- mean component count MAE delta: `0.000000`

Oracle best-per-image within R225 candidates:

- rows/images: `12`
- quick-useful: `12`
- hard-risk: `0`
- mean Dice delta: `+0.000096`
- mean IoU delta: `+0.000162`
- mean Recall delta: `-0.000142`
- mean Boundary IoU delta: `+0.001187`
- mean Boundary F1 delta: `+0.001484`
- mean gap-region FP delta: `-0.000704`
- mean component count MAE delta: `-0.083333`

## Interpretation

R225 does not recover the R224 signal.

R224 pairwise GT-instance candidates:

- `1118/1208` useful;
- `54/1208` hard-risk;
- mean cut GT foreground fraction `0.002231`;
- oracle best Boundary IoU delta `+0.003968`;
- oracle best gap FP delta `-0.002976`.

R225 pseudo-lobe candidates:

- only `260/609` useful;
- `340/609` hard-risk;
- mean cut GT foreground fraction `0.649982`;
- overall Boundary IoU is slightly worse;
- runtime is too slow for broad sweeps.

The safe-gap subset proves that good candidates still exist, but that subset is
identified by GT diagnostics. The GT-free pseudo-lobe generator itself is not
clean enough.

## Decision

`R225 = no_go_for_mask_level`.

Do not:

- run full-val parameter sweeps of the same pseudo-lobe heuristic;
- write R225 masks;
- apply R225 to clean-test-v2;
- claim any R201 improvement from R225.

## Next Route

The R224 hypothesis remains strong, but naive distance-peak pseudo-lobes are
not enough. The next route should learn or infer the pairwise seam relation more
directly.

Recommended R226 options:

1. Train a pairwise seam proposal model using R224 oracle masks:
   - input: image crop, R110 anchor component, distance maps;
   - target: R224 pairwise seam cut;
   - evaluate on original val only.
2. Build a pseudo-instance generator first:
   - predict instance/lobe seeds from R110 + image;
   - only then generate pairwise seams.
3. Use R224 positives and R225/R223 risky cuts as contrastive supervision:
   - positives: high-gap, low-foreground R224 seams;
   - negatives: pseudo-lobe interior cuts and R223 risky generic seams.

Promotion gate remains unchanged:

- original-val mask-level result;
- Dice/IoU/Recall non-degradation;
- Boundary IoU/F1 improvement;
- gap FP reduction;
- component count MAE non-worsening;
- hard-case visualization with no over-erosion.

