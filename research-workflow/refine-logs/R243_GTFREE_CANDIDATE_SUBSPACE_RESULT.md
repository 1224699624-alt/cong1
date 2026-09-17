# R243-F0 GT-Free Candidate Subspace Result

## ARIS Status
- Stage: original-val candidate-subspace diagnostic
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `no_go_r239_candidate_distribution_too_noisy`

## Purpose
R242 showed that a local crop CNN cannot safely select from the full R239
background-channel candidate distribution. R243-F0 asks a simpler question:

Can a stricter, deployable, GT-free subspace of the same candidates become clean
enough for a mask-level editor?

The search uses only GT-free columns for selection, such as:

- candidate family
- background-channel proxy
- cut background-channel fraction after deletion
- crop border contacts
- ring background/foreground fraction
- cut distance inside anchor
- cut area/fill/slenderness
- seam-probability summary
- action fraction

GT-derived columns are used only after selection for original-val audit.

## Outputs
- Script: `scripts/search_r243_gtfree_candidate_subspace.py`
- JSON: `outputs/analysis/r243_gtfree_candidate_subspace_search.json`
- CSV: `outputs/analysis/r243_gtfree_candidate_subspace_search.csv`

## Input
Source rows:

- `outputs/analysis/r242_candidate_crop_scorer_fullval_probed_rows.csv`

Input candidate table:

- rows: `8190`
- images: `96`
- safe labels: `765`
- risk labels: `7339`

## Result
No searched GT-free subspace passed the promotion gate:

- passing configs: `0`

Best strict subspace:

- family: `bg_seed`
- requires `bg_channel_proxy=1`
- requires `cut_component_bg_frac_after >= 0.6`
- requires `cut_component_border_contacts_after >= 4`
- requires `cut_mean_dist_in <= 1.25`
- requires `cut_max_dist_in <= 2.0`
- requires `cut_area <= 4`
- requires `cut_fill <= 0.1`
- requires `cut_slenderness >= 4`
- max action fraction: `0.25`

Best strict subspace audit:

| Metric | Value |
| --- | ---: |
| selected rows/images | `7/7` |
| safe rows | `1` |
| risk rows | `5` |
| safe rate | `0.142857` |
| risk rate | `0.714286` |
| Dice delta | `+0.000007` |
| IoU delta | `+0.000011` |
| Recall delta | `-0.000098` |
| Boundary IoU delta | `+0.000245` |
| Boundary F1 delta | `+0.000298` |
| gap-region FP delta | `-0.000126` |
| component count MAE delta | `+0.000000` |
| mean cut GT foreground fraction | `0.428571` |
| mean cut GT gap fraction | `0.357143` |

## Interpretation
R243-F0 is a useful negative result.

Even the most conservative GT-free subspace inside R239/R242 remains risk-heavy.
The selected rows improve boundary/gap metrics slightly, but still cut too much
GT foreground and reduce Recall. This violates the project requirement that
bone-seam separation must not be obtained through over-erosion or missed bone.

Together, R242 and R243-F0 show:

- the crop scorer is not the main blocker by itself;
- the R239 candidate distribution is too noisy;
- further threshold tuning or small classifiers over the same candidate pool
  are low value.

## Next Step
Move to a cleaner candidate generator, not another selector over R239 rows.

Recommended R244 direction:

1. Start from the R237/R238 component-merge-targeted oracle profile.
2. Generate candidates only around predicted connected components that are
   likely multi-bone adhesions using GT-free signals:
   - large/elongated component geometry;
   - multiple nearby background contacts;
   - local neck/thin-corridor evidence;
   - strong background-channel formation after a tiny cut;
   - shallow in-mask distance and high boundary support.
3. Use original-val GT only to audit whether this cleaner generator approaches
   the R238-A oracle profile:
   - more safe rows than risk rows;
   - Recall delta `0`;
   - Boundary IoU/F1 and gap FP better than R236;
   - component count MAE non-worse or improved.
4. Do not write masks or touch clean-test-v2 until the cleaner generator passes
   candidate-level and original-val mask-level gates.
