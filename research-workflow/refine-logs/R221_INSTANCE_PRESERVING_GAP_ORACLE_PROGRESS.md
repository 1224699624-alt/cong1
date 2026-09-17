# R221 Instance-Preserving Gap Oracle Progress

Date: 2026-07-08

## Purpose

Start the next route after R220: an instance-preserving seam/background
formulation for reducing bone-gap adhesion and boundary errors.

This is not a clean-test-v2 result and not a deployable model. It is a
train/val-only oracle upper-bound audit asking:

- if true bone-gap background were known,
- and we only suppressed R110 foreground pixels inside that true gap,
- could we improve gap/boundary/component diagnostics without lowering Dice,
  IoU, or Recall?

## Implementation

Added:

- `scripts/audit_r221_instance_preserving_gap_oracle.py`

The script evaluates R110 trainval anchor masks on original
`TSRS_RSNA-Epiphysis/val`. It writes diagnostics only:

- CSV: `outputs/analysis/r221_instance_preserving_gap_oracle_val32.csv`
- JSON: `outputs/analysis/r221_instance_preserving_gap_oracle_val32.json`

The audit uses GT-derived gap maps, so it is explicitly oracle evidence. It
does not write masks and does not use clean-test-v2.

## Remote Run

Because local R110 trainval val masks are incomplete, the script was synced to:

- `/home/shenzeyu/workspace/YOLO_SAM_generic_src`

Remote Python:

- `/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin/python`

Completed representative run:

- split: `val`
- limit: `32`
- missing anchor masks: `0`
- evidence level: `train_val_oracle_upper_bound`

A full 96-image val run was also launched in screen
`r221_gap_oracle_val`; it is slower because every variant computes
instance/component diagnostics. The val32 result is enough to decide whether
the route is worth pursuing, but the full result should be synced when it
finishes.

## Key Result

Decision from val32:

- `go_train_gt_free_seam_background_scorer`

Feasible oracle variants:

| Variant | Dice Δ | IoU Δ | Recall Δ | Boundary IoU Δ | Boundary F1 Δ | Gap FP Δ | Component Count MAE Δ | Merged Pred Components Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gap_between_instances_k13_maxfrac0.003` | +0.000448 | +0.000761 | +0.000000 | +0.003667 | +0.004681 | -0.002389 | -0.125000 | +0.031250 |
| `gap_between_instances_k13_maxfrac0.006` | +0.001204 | +0.002044 | +0.000000 | +0.007909 | +0.009935 | -0.006140 | -0.281250 | -0.062500 |
| `gap_between_instances_k9_maxfrac0.003` | +0.000466 | +0.000792 | +0.000000 | +0.003739 | +0.004793 | -0.002578 | -0.218750 | +0.031250 |

The strongest val32 oracle is:

- `gap_between_instances_k13_maxfrac0.006`

It improves Dice/IoU slightly, keeps Recall unchanged, improves Boundary IoU
and Boundary F1, reduces gap-region FP, reduces component count error, and
also reduces merged predicted components.

## Interpretation

R221 gives a stronger signal than R216-R218 because it changes the formulation:

- R216-R218 tried to select deletion-like seam actions from ambiguous local
  candidates.
- R221 asks for a background/seam protection map that restores true background
  between adjacent bones.
- The oracle shows that if such a map can be predicted, the desired metric
  direction is feasible without over-erosion.

This directly matches the project goal: reduce bone-seam adhesion and boundary
errors without trading them for missed bone foreground.

## Current Claim Status

Supported as internal direction evidence:

- true gap/background suppression has a positive upper bound on original val;
- improvements are not caused by recall loss in the val32 oracle;
- the best oracle variant improves both boundary and gap diagnostics.

Not yet supported:

- deployable GT-free improvement;
- full-val confirmation;
- clean-test-v2 R201 mask-level improvement;
- paper-facing claim.

## Next Step

Proceed to R222 only after syncing the full-val R221 result or deciding the
val32 result is sufficient for prototyping.

Recommended R222 route:

1. Train a GT-free seam/background scorer on original train.
2. Inputs should include image, R110 anchor, anchor boundary/distance maps, and
   local background context.
3. Target should be the R221 oracle cut map, with a strong foreground
   preservation loss.
4. Apply on original val only.
5. Accept a branch only if Dice/IoU/Recall do not drop and Boundary IoU,
   Boundary F1, gap FP, and component diagnostics improve.

Do not use clean-test-v2 for this training, threshold search, or model
selection.
