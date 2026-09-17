# R246-F0 GT-Free Merge Feature Failure Audit

## ARIS Status
- Stage: original-val feature failure audit
- Dataset: `TSRS_RSNA-Epiphysis`
- Input: R244 30/96 partial candidate table
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `no_go_simple_scalar_prefilter`

## Purpose
R245 showed that the current GT-free hard-filter/top1 route does not generalize
from 10/96 to 20/96 or 30/96. R246 audits why the GT-free filter fails by
comparing:

- all R244 candidates
- safe-useful rows
- risk rows
- oracle-safe best rows
- rows selected by the best R245 partial30 rule

GT labels are used only for grouping and audit, not for deployable selection.

## Outputs
- Script: `scripts/audit_r246_gtfree_merge_feature_failure.py`
- JSON: `outputs/analysis/r246_gtfree_merge_feature_failure.partial30.json`
- CSV: `outputs/analysis/r246_gtfree_merge_feature_failure_feature_table.partial30.csv`
- Ad-hoc component-preselection table:
  `outputs/analysis/r247_component_prescreen_ad_hoc.partial30.csv`

## Main Numbers

R244 oracle-safe best at 30/96:

- rows/images: `27/27`
- safe/risk: `27/0`
- Dice delta `+0.000235`
- IoU delta `+0.000387`
- Recall delta `+0.000000`
- Boundary IoU delta `+0.001376`
- Boundary F1 delta `+0.001731`
- gap-region FP delta `-0.001057`
- component count MAE delta `-0.481481`
- mean cut GT foreground fraction `0`

R245 best-selected at 30/96:

- rows/images: `5/5`
- safe/risk: `3/3`
- Dice delta `+0.000065`
- IoU delta `+0.000104`
- Recall delta `-0.000071`
- Boundary IoU delta `-0.000025`
- Boundary F1 delta `-0.000059`
- gap-region FP delta `-0.000455`
- mean cut GT foreground fraction `0.366667`

## Feature Observations
Compared with oracle-safe best rows, R245-selected rows tend to come from:

- much smaller predicted components:
  - oracle component area mean `11966.7`
  - R245-selected component area mean `5920.2`
- lower distance percentile:
  - oracle mean `10.22`
  - R245-selected mean `8.00`
- later candidate rank:
  - oracle mean `11.44`
  - R245-selected mean `16.40`

This suggests that the failure is not only at cut-level geometry. The pipeline
also needs better component-level targeting: which predicted connected component
is likely a true adjacent-bone adhesion worth cutting?

## R247 Ad-Hoc Component Prefilter
A temporary component-preselection search tested simple deployable thresholds
over:

- minimum component area
- minimum distance percentile
- maximum candidate rank
- background-channel contacts/fraction
- cut area and distance constraints

No searched configuration passed the promotion gate on partial30.

Best ad-hoc row:

- rows/images: `13/13`
- safe/risk: `8/6`
- Dice delta `+0.000030`
- IoU delta `+0.000049`
- Recall delta `-0.000029`
- Boundary IoU delta `+0.000117`
- Boundary F1 delta `+0.000161`
- gap-region FP delta `-0.000207`
- mean cut GT foreground fraction `0.221154`
- decision: no-go

## Interpretation
The good news is that R244's oracle-safe rows are strong and anatomically
aligned. The bad news is that simple scalar GT-free thresholds cannot reliably
recover them.

Do not promote R245/R247 to mask writing. The next useful route should revise
candidate generation or ranking with richer context, for example:

1. component-level adhesion suspicion before generating cuts;
2. local crop/context scoring trained on R244/R237 oracle-safe rows and hard
   negatives from R245/R247;
3. a generator that proposes fewer cuts per component, closer to the R237
   oracle geometry;
4. a hard rule that forbids any clean-test-v2 use until original-val candidate
   gates pass.
