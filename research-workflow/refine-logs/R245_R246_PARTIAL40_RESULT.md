# R245/R246 Partial40 Result

## ARIS Status
- Stage: R244 partial40 original-val audit
- Dataset: `TSRS_RSNA-Epiphysis`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `no_go_scalar_filter_confirmed_partial40`

## Inputs
- R244 summary: `outputs/analysis/r244_gtfree_merge_targeted_fullval_summary.partial40.json`
- R244 candidates: `outputs/analysis/r244_gtfree_merge_targeted_fullval_candidates.partial40.csv`
- R245 search: `outputs/analysis/r245_gtfree_merge_filter_search.partial40.top1.json`
- R246 audit: `outputs/analysis/r246_gtfree_merge_feature_failure.partial40.json`

## R244 Partial40 Signal
R244 now covers:

- candidate rows/images: `3533/40`
- safe-useful rows: `464`
- risk rows: `3086`
- deployable clean subset: `36` safe vs `201` risk

Oracle-safe best remains strong:

- rows/images: `36/36`
- safe/risk: `36/0`
- Dice delta `+0.000211`
- IoU delta `+0.000349`
- Recall delta `+0.000000`
- Boundary IoU delta `+0.001240`
- Boundary F1 delta `+0.001619`
- gap-region FP delta `-0.001087`
- component count MAE delta `-0.777778`
- mean cut GT foreground fraction `0`
- mean cut GT gap fraction `0.996944`

Interpretation: the reversed-background-connectivity upper bound remains alive
and increasingly relevant to component-count diagnostics.

## R245 Partial40 Result
Best GT-free top1 scalar filter:

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

The best scalar GT-free filter still violates the project requirement: seam
separation must not be obtained through foreground erosion or recall loss.

## R246 Partial40 Failure Audit
R246 confirms the gap between the oracle and deployable selection:

- `oracle_safe_best`: `36` safe, `0` risk, Recall `0`, Boundary IoU `+0.001240`
- `r245_best_selected`: `3` safe, `3` risk, Recall `-0.000071`, Boundary IoU `-0.000025`

This is not a threshold-tuning problem anymore. The useful candidates exist,
but the scalar GT-free selection cannot reliably find them.

## Next Step
Keep R244 running to full original-val completion. R249 is already waiting and
will run full R245/R246 automatically after R244 exits.

Do not write masks from R245. If full R245/R246 remains no-go, launch the
prepared R248 context-crop scorer, which can inspect local image/cut/component
context instead of relying only on scalar features.
