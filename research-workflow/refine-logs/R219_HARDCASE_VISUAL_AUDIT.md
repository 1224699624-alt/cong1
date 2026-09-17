# R219 Hard-Case Visual Audit

Date: 2026-07-08

## Purpose

R216-R218 showed that soft seam actions have oracle value, but action gates do
not generalize safely. R219 creates visual audit artifacts so the next model
direction is based on inspected failure modes rather than another blind gate
variant.

This is diagnostic work only.

- clean-test-v2 is used only through the existing R203/R206 diagnostic
  manifest and visualization pack;
- R216/R218 action visualizations use original validation data only;
- no model masks are written.

## Existing Clean-Test Diagnostic Pack

Existing artifact:

- `outputs/analysis/r206_hardcase_visualizations/index.html`

Source:

- R203 clean-test-v2 failure manifest;
- R201 metrics for ARAA, R110, and R202 nnU-Net risk-audit baseline.

Purpose:

- inspect where R110 has bridge/gap/component failures relative to ARAA/R202;
- verify qualitative target cases for the project claim.

Important caution:

- These clean-test-v2 cases must not be used for threshold tuning or model
  selection.

## New Action-Level Diagnostic Pack

Implemented:

- `scripts/build_r219_action_hardcase_visualizations.py`

Generated artifact:

- `outputs/analysis/r219_action_hardcase_visualizations/index.html`
- `outputs/analysis/r219_action_hardcase_visualizations/r219_action_hardcase_index.csv`
- `outputs/analysis/r219_action_hardcase_visualizations/r219_action_hardcase_summary.json`
- `outputs/analysis/r219_action_hardcase_visualizations/panels/*.jpg`

Summary:

- total cases: `24`
- safe useful: `8`
- hard risk: `8`
- ambiguous: `8`

Groups:

- `safe_useful`: actions that reduce gap/boundary errors with low diagnostic
  risk.
- `hard_risk`: actions that cut too much foreground / over-erode bone.
- `ambiguous`: actions with useful-looking metrics but foreground-overlap or
  local geometry that explains why gates confuse them.

Representative rows:

- safe useful: `1620.png`, `11852.png`, `1666.png`, `13436.png`, `15067.png`
- hard risk: `1934.png`, `1855.png`, `1560.png`, `1452.png`, `5825.png`
- ambiguous: `1694.png`, `13441.png`, `10375.png`, `1886.png`

## Audit Interpretation

The visual pack supports three working conclusions:

1. Partial seam actions are real: some candidate cuts visibly remove bridge/gap
   false positives and improve local boundary alignment.
2. Many risk cases are not obvious from simple local geometry: harmful cuts can
   look seam-like but remove true foreground, explaining why R216/R217/R218
   gates fail under grouped CV.
3. The current action formulation is still deletion-like. Even when partial,
   it can trade bridge reduction for foreground erosion unless instance or
   anatomical preservation is built into the action itself.

What R219 does not support:

- It does not prove a new model improves R201 metrics.
- It does not justify applying R216/R217/R218 to clean-test-v2.
- It does not justify more threshold sweeps on the same candidate/action table.

## Decision

Do not continue with more R216-R218 gate variants unless the candidate/action
formulation changes.

Recommended next direction:

1. Build an instance-preserving seam/background formulation:
   - predict a seam/background probability or protection map;
   - constrain edits so component count error cannot worsen;
   - prefer suppression inside predicted gap channels rather than deleting
     arbitrary anchor foreground.
2. Use R219 panels to select train/val qualitative sanity cases for debugging,
   not for clean-test-v2 tuning.
3. In parallel, continue representative baseline completion / comparison, since
   the current method branch has not yet produced mask-level improvement.

