# R249/R250 Full Chain Status

## ARIS Status
- Decision: `r248_no_go_context_scorer_not_clean_enough`
- Clean-test-v2 used: `false`
- Writes masks: `false`
- Scope: `TSRS_RSNA-Epiphysis original-val diagnostics only`

## Remote/Artifact Status
- R244 running: `false`
- R249 waiting: `false`
- R250 waiting: `false`
- R248 running: `false`
- R244 log progress: `None/None`
- R249 seen summary: `90 images / 8366 rows`
- Full R245 exists: `true`
- Full R246 exists: `true`
- Full R248 exists: `true`

## R244 Oracle/GT-Free Snapshot
- candidate images: `96.000000000`
- candidate rows: `8568.000000000`
- oracle images: `87.000000000`
- oracle Boundary IoU delta: `0.001281313`
- oracle gap FP delta: `-0.001125485`
- oracle component count MAE delta: `-0.563218391`
- GT-free clean safe/risk: `79.000000000` / `476.000000000`

## R245 Gate
- R245 decision field: `no_go_filter_not_clean_enough`
- R245 passing configs: `0.000000000`
- best num_rows: `9.000000000`
- best num_images: `9.000000000`
- best num_safe_useful: `6.000000000`
- best num_risk: `4.000000000`
- best mean_delta_dice: `0.000155649`
- best mean_delta_iou: `0.000256016`
- best mean_delta_recall: `-0.000053479`
- best mean_delta_boundary_iou: `0.000847426`
- best mean_delta_boundary_f1: `0.001092563`
- best mean_delta_gap_region_fp_rate: `-0.000833716`
- best mean_delta_component_count_mae: `0.000000000`
- best mean_cut_gt_fg_frac: `0.184523810`
- best mean_cut_gt_gap_frac: `0.783658009`

Passed checks:
- selected: num_rows=9
- safe_useful: safe=6 >= 5
- boundary_iou_gain: boundary_iou_delta=0.000847426 >= 0.0001
- boundary_f1_gain: boundary_f1_delta=0.001092563
- gap_fp_gain: gap_fp_delta=-0.000833716
- low_foreground_cut: cut_gt_fg_frac=0.184523810

Failed checks:
- risk: risk=4 <= 2
- recall_nonworse: recall_delta=-0.000053479

## R246 Feature-Failure Groups
- oracle_safe_best:
  - num_rows: `87.000000000`
  - num_images: `87.000000000`
  - num_safe_useful: `87.000000000`
  - num_risk: `0.000000000`
  - mean_delta_recall: `0.000000000`
  - mean_delta_boundary_iou: `0.001281313`
  - mean_delta_boundary_f1: `0.001695222`
  - mean_delta_gap_region_fp_rate: `-0.001125485`
  - mean_delta_component_count_mae: `-0.563218391`
  - mean_cut_gt_fg_frac: `0.000000000`
  - mean_cut_gt_gap_frac: `0.998053719`
- r245_best_selected:
  - num_rows: `9.000000000`
  - num_images: `9.000000000`
  - num_safe_useful: `6.000000000`
  - num_risk: `4.000000000`
  - mean_delta_recall: `-0.000053479`
  - mean_delta_boundary_iou: `0.000847426`
  - mean_delta_boundary_f1: `0.001092563`
  - mean_delta_gap_region_fp_rate: `-0.000833716`
  - mean_delta_component_count_mae: `0.000000000`
  - mean_cut_gt_fg_frac: `0.184523810`
  - mean_cut_gt_gap_frac: `0.783658009`
- safe_useful:
  - num_rows: `971.000000000`
  - num_images: `87.000000000`
  - num_safe_useful: `971.000000000`
  - num_risk: `124.000000000`
  - mean_delta_recall: `0.000000000`
  - mean_delta_boundary_iou: `0.000532049`
  - mean_delta_boundary_f1: `0.000727306`
  - mean_delta_gap_region_fp_rate: `-0.000673686`
  - mean_delta_component_count_mae: `-0.165808445`
  - mean_cut_gt_fg_frac: `0.000000000`
  - mean_cut_gt_gap_frac: `0.980571672`
- risk:
  - num_rows: `7590.000000000`
  - num_images: `95.000000000`
  - num_safe_useful: `124.000000000`
  - num_risk: `7590.000000000`
  - mean_delta_recall: `-0.000269025`
  - mean_delta_boundary_iou: `-0.000092016`
  - mean_delta_boundary_f1: `-0.000119455`
  - mean_delta_gap_region_fp_rate: `-0.000347574`
  - mean_delta_component_count_mae: `0.151515152`
  - mean_cut_gt_fg_frac: `0.707375985`
  - mean_cut_gt_gap_frac: `0.245424423`

## R248 Context-Crop Scorer
- best: `missing or null`
- original grid passing configs: `0.000000000`
- OOF risk probability range: `0.702876985` to `0.999949098`
- reachable regrid nonempty configs: `68.000000000`
- reachable regrid passing configs: `0.000000000`
- reachable regrid decision: `no_go_context_scorer_not_clean_enough`
- regrid best useful_threshold: `0.840000000`
- regrid best risk_threshold: `0.760000000`
- regrid best selected: `4.000000000`
- regrid best selected_images: `4.000000000`
- regrid best safe_useful: `2.000000000`
- regrid best risk: `2.000000000`
- regrid best mean_delta_recall: `-0.000015604`
- regrid best mean_delta_boundary_iou: `0.000255240`
- regrid best mean_delta_boundary_f1: `0.000321331`
- regrid best mean_delta_gap_region_fp_rate: `-0.000234329`
- regrid best mean_delta_component_count_mae: `0.000000000`
- regrid best mean_cut_gt_fg_frac: `0.218750000`

## Interpretation
- If decision is `waiting_for_r244_fullval`, do not launch additional CPU-heavy jobs.
- If R245 passes, next step is an isolated original-val mask editor and R201-style audit, not clean-test-v2.
- If R245 remains no-go and R248 is missing, wait for R250 or launch R248 manually only after confirming R250 failed.
- The original R248 risk-threshold grid ended at `0.50`, below every OOF risk probability; `best=null` alone is not a valid model conclusion.
- The reachable R248b regrid is the corrected gate. If it remains no-go, move to explicit seam/background supervision or reduce generator risk before any mask writing.
- clean-test-v2 remains locked for final evaluation/diagnosis only.
