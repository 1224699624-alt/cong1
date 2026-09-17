# R224 Pairwise Instance-Seam Candidate Result

Date: 2026-07-08

## Purpose

R223 showed that generic structured seam/background proposals contain useful
signal, but GT-free selectors still confuse useful seam cuts with over-erosion
risk. R224 tests a cleaner candidate-generation hypothesis:

- only generate seam candidates inside an R110 anchor component that overlaps
  multiple GT instances;
- propose cuts between a specific pair of neighboring GT instances;
- evaluate whether such pairwise seam candidates reduce bone-gap adhesion and
  boundary errors without cutting true bone foreground.

This is original-val oracle diagnostic evidence only. GT instances are used to
generate candidates, so this is not a deployable method, not mask-level R201
evidence, and not clean-test-v2 evidence.

## Implemented

Added:

- `scripts/audit_r224_pairwise_instance_seam_candidates.py`

The script:

- reads R110 train/val anchor masks;
- finds anchor connected components overlapping multiple GT instances;
- creates pairwise seam candidates between GT-instance pairs;
- trims candidates using shallow-boundary / gradient-aware scoring;
- evaluates Dice, IoU, Recall, Boundary IoU/F1, gap FP, and component-count
  diagnostics on original val;
- writes JSON/CSV only.

Artifacts:

- `outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv`
- `outputs/analysis/r224_pairwise_instance_seam_fullval_summary.json`
- `outputs/logs/r224_pairwise_instance_seam_fullval.log`

## Full-Val Result

Settings:

- split: original `TSRS_RSNA-Epiphysis/val`
- clean-test-v2 used: `false`
- writes masks: `false`
- seam radii: `2,3,4`
- action fractions: `0.25,0.50,0.75`
- max pairs per component: `4`

Coverage:

- candidate rows: `1208`
- images with candidates: `70`

Overall candidates:

- quick-useful: `1118/1208`
- hard-risk: `54/1208`
- safe-gap positive: `1203/1208`
- over-erosion proxy: `1/1208`
- mean Dice delta: `+0.000162`
- mean IoU delta: `+0.000274`
- mean Recall delta: `-0.000001`
- mean Boundary IoU delta: `+0.001360`
- mean Boundary F1 delta: `+0.001817`
- mean gap-region FP delta: `-0.000937`
- mean component count MAE delta: `+0.009934`
- mean cut GT foreground fraction: `0.002231`
- mean cut GT gap fraction: `0.997769`

Safe-gap positive candidates:

- rows: `1203`
- quick-useful: `1113`
- hard-risk: `54`
- over-erosion proxy: `0`
- mean Boundary IoU delta: `+0.001362`
- mean Boundary F1 delta: `+0.001819`
- mean gap-region FP delta: `-0.000937`
- mean cut GT foreground fraction: `0.000323`
- mean cut GT gap fraction: `0.999677`

Oracle best-per-image:

- rows/images: `68`
- quick-useful: `68`
- hard-risk: `0`
- mean Dice delta: `+0.000517`
- mean IoU delta: `+0.000868`
- mean Recall delta: `-0.000006`
- mean Boundary IoU delta: `+0.003968`
- mean Boundary F1 delta: `+0.005319`
- mean gap-region FP delta: `-0.002976`
- mean component count MAE delta: `-0.058824`
- mean cut GT foreground fraction: `0.010239`
- mean cut GT gap fraction: `0.989761`

## Action Fraction Analysis

Mean deltas by action fraction:

- `0.25`:
  - quick-useful rate: `0.903`
  - hard-risk rate: `0.074`
  - Boundary IoU delta: `+0.000615`
  - gap FP delta: `-0.000469`
- `0.50`:
  - quick-useful rate: `0.931`
  - hard-risk rate: `0.035`
  - Boundary IoU delta: `+0.001338`
  - gap FP delta: `-0.000935`
- `0.75`:
  - quick-useful rate: `0.943`
  - hard-risk rate: `0.025`
  - Boundary IoU delta: `+0.002131`
  - gap FP delta: `-0.001408`

Unlike earlier hard-deletion candidates, larger pairwise seam cuts remain safe
because the candidate is already localized to true inter-instance gap.

## Interpretation

R224 is the strongest direction so far for the current goal.

Compared with R223:

- R223 generic proposals had `216/530` useful and `284/530` hard-risk.
- R224 pairwise oracle proposals have `1118/1208` useful and only `54/1208`
  hard-risk.
- R223 oracle best-per-image Boundary IoU delta was `+0.001017`.
- R224 oracle best-per-image Boundary IoU delta is `+0.003968`.
- R224 also improves gap FP and component count MAE while preserving Recall.

This strongly supports the revised hypothesis:

The problem is not just "find seam-like pixels"; it is "identify which adjacent
bone-instance pair a seam belongs to, then open the background channel between
that pair."

## Decision

`R224 = go_for_gt_free_pairwise_seam_generator`.

Do not:

- apply R224 directly to clean-test-v2;
- claim R201 improvement from R224;
- write final masks using GT-instance candidates.

Do:

- proceed to R225, replacing GT-instance candidate generation with a GT-free or
  pseudo-instance pairwise seam generator.

## Recommended R225

R225 should target deployable candidate generation, not another selector over
generic cuts.

First R225 options:

1. Build pseudo-instance pairs from R110 anchor geometry:
   - split large merged anchor components into lobes using distance-transform
     ridges, skeleton junctions, or watershed seeds;
   - generate candidate seams only between pseudo-lobe pairs.
2. Train a pairwise seam proposal model:
   - input: image crop, anchor component, distance maps, and two-side lobe
     masks;
   - target: R224 pairwise oracle seam mask;
   - evaluate on original val only.
3. Use R224 rows as supervision:
   - positives are pairwise cuts with high gap fraction and useful deltas;
   - negatives are non-pairwise/generic risky cuts or same-component interior
     cuts.

Promotion gate for R225:

- mask-level original-val result;
- Dice/IoU/Recall non-degradation;
- Boundary IoU/F1 improvement;
- gap-region FP reduction;
- component count MAE non-worsening;
- hard-case visualization showing reduced bone seam adhesion without
  over-erosion.

