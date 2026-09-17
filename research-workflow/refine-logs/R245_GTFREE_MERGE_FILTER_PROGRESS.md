# R245-F0 GT-Free Merge Filter Progress

## ARIS Status
- Stage: original-val GT-free filter/search diagnostic
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: R110 val masks
- Clean-test-v2 used: no
- Writes masks: no
- Decision so far: `continue_wait_full_r244_but_do_not_promote`

## Purpose
R244-F0 tests the reversed-topology route: encourage/connect background through
bone seams so adjacent epiphysis regions separate, without using GT at
deployment time.

R245-F0 searches deployable GT-free filters and ranking rules over the R244
candidate rows. Selection uses only image/anchor/candidate geometry features.
GT-derived columns are used only for original-val audit.

## Added Script
- `scripts/search_r245_gtfree_merge_filter.py`

The script reports:
- `num_rows`, `num_images`
- `num_safe_useful`, `num_risk`
- Dice/IoU/Recall/Boundary/gap/component deltas
- `clean_test_v2_used=false`
- `writes_masks=false`
- explicit marker that GT columns are audit-only

## R244 Remote Monitor
Remote screen:

- `r244_gtfree_merge_fullval`

At the 20/96 flush:

- candidate rows/images: `841/20`
- safe-useful rows: `150`
- risk rows: `693`
- initial R244 `gtfree_clean_subset`: `15 safe / 79 risk`, no-go
- oracle-safe best over `19` images:
  - Dice delta `+0.000254`
  - IoU delta `+0.000416`
  - Recall delta `+0.000000`
  - Boundary IoU delta `+0.001516`
  - Boundary F1 delta `+0.001888`
  - gap-region FP delta `-0.001057`
  - component count MAE delta `-0.315789`
  - mean cut GT foreground fraction `0`
  - mean cut GT gap fraction `1`

Interpretation: the oracle signal remains alive and stronger than R236, but the
deployable R244 filter is still too risky.

## R245 Partial Audits

### Partial 10/96
Input:
- `outputs/analysis/r244_gtfree_merge_targeted_fullval_candidates.partial.csv`

Best top-1 GT-free filter:
- selected rows/images: `3/3`
- safe/risk: `3/1`
- Dice delta `+0.000167`
- IoU delta `+0.000267`
- Recall delta `+0.000000`
- Boundary IoU delta `+0.000144`
- Boundary F1 delta `+0.000189`
- gap-region FP delta `-0.000725`
- mean cut GT foreground fraction `0`
- mean cut GT gap fraction `0.904545`
- decision: `go_next_original_val_mask_editor` on this small partial only

This was useful because it exposed a script bug where `mean_cut_gt_fg_frac=0`
was incorrectly treated as missing in the promotion gate. The bug was fixed.

### Partial 20/96
Input:
- `outputs/analysis/r244_gtfree_merge_targeted_fullval_candidates.partial20.csv`

Best top-1 GT-free filter:
- selected rows/images: `16/16`
- safe/risk: `8/7`
- Dice delta `+0.000076`
- IoU delta `+0.000119`
- Recall delta `-0.000084`
- Boundary IoU delta `+0.000536`
- Boundary F1 delta `+0.000676`
- gap-region FP delta `-0.000394`
- mean cut GT foreground fraction `0.240057`
- mean cut GT gap fraction `0.572443`
- decision: `no_go_filter_not_clean_enough`

Interpretation: the partial-10 rule does not generalize cleanly to 20 images.
Boundary/gap metrics move in the desired direction, but part of that movement is
still obtained by cutting true foreground. This violates the current project
requirement.

### Partial 30/96
Input:
- `outputs/analysis/r244_gtfree_merge_targeted_fullval_candidates.partial30.csv`

Best top-1 GT-free filter:
- selected rows/images: `5/5`
- safe/risk: `3/3`
- Dice delta `+0.000065`
- IoU delta `+0.000104`
- Recall delta `-0.000071`
- Boundary IoU delta `-0.000025`
- Boundary F1 delta `-0.000059`
- gap-region FP delta `-0.000455`
- mean cut GT foreground fraction `0.366667`
- mean cut GT gap fraction `0.576061`
- decision: `no_go_filter_not_clean_enough`

Interpretation: the top-1 hard-filter route is not stable as more images enter
the audit. Unlike the 20/96 flush, the best 30/96 rule no longer improves
boundary metrics and cuts too much true foreground. This blocks promotion to any
mask-writing editor.

## Current Conclusion
The reversed-background-connectivity route remains promising because R244's
oracle-safe candidates are still strong and anatomically aligned. However, the
current deployable GT-free filter is not stable enough for mask writing. The
partial trend from 10/96 to 30/96 argues against pushing the present R245 filter
into a mask-level editor.

Do not promote R245 to original-val mask-level editing until the full R244 run
is complete and a full-val GT-free filter passes:

- safe rows greater than risk rows
- Recall delta non-negative
- Boundary IoU/F1 improved
- gap-region FP reduced
- low mean cut GT foreground fraction
- no clean-test-v2 use

## Next Step
Wait for R244 full original-val completion, sync the final candidate table, then
rerun R245 on all `96` val images for a final audit. If the full R245 result
remains risk-heavy, revise candidate generation/ranking rather than writing
masks:

1. penalize overly strong background channels that often indicate foreground
   cuts;
2. favor shallow, low-gradient, compact cuts;
3. consider component-level preselection before candidate generation;
4. keep GT strictly audit-only until a candidate-level gate passes.
