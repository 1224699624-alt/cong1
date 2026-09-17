# R240 Geometry Gate Audit Result

## ARIS Status
- Stage: original-val GT-free gate audit
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `no_go_geometry_gate_too_risky; train_local_gap_scorer_next`

## Purpose
R237/R238 showed that the reversed-topology idea is valid under oracle
candidate generation: opening a small background channel in the bone seam can
reduce gap false positives, improve boundary metrics, and sometimes improve
component-count diagnostics without recall loss.

R239 then tried a deployable GT-free version using background seeds plus a
seam-probability hard gate, but it was too conservative (`2/96` edited).

R240 asks a narrower question:

- can GT-free background-channel geometry replace the overly conservative
  seam-probability gate?
- can we select many more seam cuts while still avoiding over-erosion or漏分?

## Implementation
Added:

- `scripts/run_r240_bg_channel_geometry_editor.py`
- `scripts/audit_r240_geometry_gate_from_candidates.py`

The mask editor keeps R239 candidate generation but changes the intended
selection policy: background-channel geometry is the hard filter, while seam
probability is only a ranking feature. Because full mask generation was slow
under the broader geometry gate, the decisive R240 result uses the already
generated R239 candidate table:

- `outputs/analysis/r239_gtfree_bg_channel_light_fullval_candidates.csv`

Selection uses only GT-free columns such as:

- `bg_channel_proxy`
- `cut_component_bg_frac_after`
- `ring_bg_frac`
- `cut_mean_dist_in`
- `cut_max_dist_in`
- `cut_area`
- `bridge_score_p90`

GT-derived columns are used only after selection for original-val audit.

## Main Result
Default geometry gate:

- selected rows/images: `54/54`
- safe gains: `12`
- risk rows: `42`
- Dice delta: `-0.000039`
- IoU delta: `-0.000055`
- Recall delta: `-0.000185`
- Boundary IoU delta: `+0.000107`
- Boundary F1 delta: `+0.000116`
- gap FP delta: `-0.000230`
- component count MAE delta: `-0.018519`
- mean cut GT foreground fraction: `0.538601`
- mean cut GT gap fraction: `0.425504`

Strict geometry gate:

- selected rows/images: `39/39`
- safe gains: `7`
- risk rows: `32`
- Dice delta: `-0.000037`
- IoU delta: `-0.000053`
- Recall delta: `-0.000174`
- Boundary IoU delta: `+0.000088`
- Boundary F1 delta: `+0.000102`
- gap FP delta: `-0.000177`
- component count MAE delta: `-0.025641`
- mean cut GT foreground fraction: `0.564031`
- mean cut GT gap fraction: `0.364082`

## Interpretation
R240 is a useful negative result.

The background-channel geometry gate does select edits that move the target
diagnostics in the desired direction: Boundary IoU/F1 improve, gap FP decreases,
and component count MAE slightly improves. However, it does this by frequently
cutting true GT foreground. The mean `cut_gt_fg_frac` is above `0.5` in both the
default and strict gates, and risk rows heavily outnumber safe rows.

Therefore, pure hand-written geometry is not sufficient. This route cannot be
promoted to mask-level clean-test-v2 evaluation.

## Comparison With Nearby Runs
| Run | Evidence | Edited / selected | Safety | Boundary / gap effect | Decision |
| --- | --- | ---: | --- | --- | --- |
| R236 | original-val mask-level | `20/96` edited | safe | weak positive | best safe coverage so far |
| R238-A | oracle mask-level | `61/96` edited | oracle-safe | strong positive | upper bound only |
| R239 | GT-free mask-level | `2/96` edited | safe | tiny positive | too conservative |
| R240 | GT-free gate audit | `54` selected | too risky | positive but unsafe | no-go |

## Next Step: R241
The next experiment should not be another threshold sweep. The blocker is
semantic discrimination:

- safe seam/background cuts and risky shallow bone cuts look similar under
  simple geometry;
- seam probability is too conservative and misses many safe gap cuts;
- geometry alone is too permissive and cuts bone.

R241 should train a local seam-background scorer from original train/val only.
Suggested target:

- positives: R237/R238-A safe pure-gap cuts and R239 safe candidates;
- hard negatives: R240 selected risky cuts with high `cut_gt_fg_frac`, recall
  loss, or boundary degradation;
- inputs: local image crop, anchor mask crop, candidate cut mask, distance
  transforms, background-channel map, and optional seam-probability map;
- grouped image-level CV before any mask writing;
- promotion gate: risk rows must be far below safe rows, Recall delta must stay
  at `0`, and mask-level original-val result must beat R236 on boundary/gap
  without over-erosion.

Do not use clean-test-v2 until R241 or a later non-oracle method passes the
original-val safety and effect-size gates.
