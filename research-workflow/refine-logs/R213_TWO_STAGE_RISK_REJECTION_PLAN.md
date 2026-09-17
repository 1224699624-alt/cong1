# R213 Two-Stage Risk-Rejection Acceptance Gate Plan

Date: 2026-07-07

## Purpose

R212-F5 showed that simple threshold/static-filter tuning around the current
candidate acceptor is exhausted. Safe settings become near no-op, while
useful-recall settings admit many risk edits. R213 should therefore test a
two-stage acceptance gate:

1. recover useful candidate recall;
2. reject risky candidates before mask application.

This is still a validation-only development route. `clean-test-v2` remains
locked until original full-val mask-derived R201 metrics and visual audit pass.

## Evidence Triggering R213

R212-F5 eval-only full original-val:

- selected threshold `0.80`;
- accepted only `1/96` image;
- Boundary IoU delta `+0.00000248`;
- Boundary F1 delta `+0.00000334`;
- Surface Dice 2px delta `+0.00000647`;
- decision from `audit_r213_validation_gate.py`: `do_not_promote`.

Candidate-signal audit:

- full-val candidate CSV contains `75` strict useful candidates across `41`
  images;
- F5 static filters leave only `6` useful candidates;
- F5 probability model scores `63/75` strict useful candidates below `0.50`.

Candidate-policy frontier:

- `risk <= 0`: at most `1` strict useful accepted;
- `risk <= 1`: at most `2` strict useful accepted;
- reaching `10-12` strict useful accepts requires `13-14` risk accepts;
- reaching `15` strict useful accepts requires about `24` risk accepts.

Conclusion: threshold/filter search alone cannot solve the acceptance problem.

## R213 Hypothesis

A two-stage candidate policy can improve useful candidate recall while keeping
hard-risk candidates out:

- Stage A: useful-recall model predicts candidates with positive boundary/gap
  effect under small Dice/IoU/Recall tolerance.
- Stage B: risk-rejection model vetoes candidates predicted to cause hard
  damage: Dice/IoU/Recall beyond tolerance or negative boundary/surface delta.

The final image-level application still uses the component-preserving guard:

- no predicted component-count increase;
- no step-wise component-count increase;
- bounded cut fraction;
- full R201 validation gate.

## Data Protocol

Allowed:

- original `TSRS_RSNA-Epiphysis` train/val candidate CSVs;
- R110 train/val anchor masks;
- validation-only candidate labels and mask-derived metrics for model selection.

Forbidden:

- no `clean-test-v2` tuning;
- no `TSRS_RSNA-Articular-Surface`;
- no raw dataset modification.

## Candidate Labels

Strict useful candidate:

- `delta_dice >= -5e-4`;
- `delta_iou >= -8e-4`;
- `delta_recall >= -8e-4`;
- `delta_boundary_iou >= 1e-4`;
- `delta_boundary_f1 >= 1e-4`;
- `delta_surface_dice_2px >= 1e-4`;
- `delta_gap_region_fp_rate < 0`;
- `delta_component_count_mae <= 0`.

Hard-risk candidate:

- `delta_dice < -5e-4`; or
- `delta_iou < -8e-4`; or
- `delta_recall < -8e-4`; or
- `delta_boundary_iou < 0`; or
- `delta_boundary_f1 < 0`; or
- `delta_surface_dice_2px < 0`.

## Experiment Steps

M0. Wait for strict R212-F5 full-train/full-val to finish, because it will
materialize the full train candidate CSV.

M1. Run candidate-level two-stage training:

- train useful model on train candidates;
- train risk model on train candidates;
- evaluate threshold grid on original val candidate CSV;
- report useful/risk frontier and selected candidate policy.

M2. If candidate-level val policy is promising, run full mask-derived original
val evaluation:

- apply two-stage policy to R110 train/val anchor masks;
- compute R201 metrics;
- run `audit_r213_validation_gate.py`;
- no clean-test-v2.

M3. If full-val mask evaluation passes, generate hard-case visual audit:

- useful accepted cases;
- rejected risk cases;
- boundary/gap before-after panels;
- over-erosion and missed-bone checks.

M4. Only after M2+M3 pass, consider one locked clean-test-v2 evaluation.

## Promotion Gate

R213 must pass:

- accepted rate `>= 0.03`;
- Dice delta `>= -5e-4`;
- IoU delta `>= -8e-4`;
- Recall delta `>= -8e-4`;
- Boundary IoU delta `>= 1e-4`;
- Boundary F1 delta `>= 1e-4`;
- Surface Dice 2px delta `>= 1e-4`;
- gap FP improves;
- component merge/count non-worse;
- no known risk-image accepts in the selected validation threshold.

## Immediate Next Action

Prepare the candidate-level two-stage audit script now. Run it first on the
available eval-only full-val candidate CSV as a wiring check, then rerun with
strict full-train/full-val train candidates once the running job completes.
