# R205-F0 Safe-Edit Feasibility Audit Result

Date: 2026-07-06

## Purpose

Before launching another anatomy-aware learned refiner, R205-F0 tested whether true R110 train/val masks contain enough safe, nonzero local correction signal.

This audit used only original `TSRS_RSNA-Epiphysis/val` labels and R110 train/val anchor masks. It did not use clean-test-v2.

## Artifacts

- Script: `scripts/run_r205_feasibility_safe_edit_audit.py`
- Smoke output: `outputs/analysis/r205_feasibility_safe_edit_audit_smoke_val8.json`
- Full val output: `outputs/analysis/r205_feasibility_safe_edit_audit_val.json`
- Full val rows: `outputs/analysis/r205_feasibility_safe_edit_audit_val.csv`

## Full Validation Result

Variant tested: `oracle_delete_gap_fp`.

This is a GT-guided upper-bound edit that deletes R110 foreground lying in the R201 GT-adjacent gap region. It is not a deployable method; it is a feasibility test.

Mean val deltas vs R110 anchor:

- Dice: `+0.036974`
- Recall: `+0.000000`
- Boundary IoU: `+0.241213`
- Boundary F1: `+0.267750`
- gap-region FP rate: `-0.188199`
- component merge rate: `-0.312500`
- component count MAE: `+22.562500`
- edited fraction: `0.001969`
- removed fraction of anchor: `0.078576`

Gate details:

- safe image count: `15/96`
- safe image rate: `0.15625`
- diagnostic improved: `true`
- mean component safe: `false`
- passes feasibility gate: `false`
- decision: `no_go_do_not_train_r205`

## Interpretation

The oracle edit shows that removing gap false positives can dramatically improve overlap, boundary, surface, and gap metrics. However, it also fragments the masks severely: component count MAE worsens by `+22.5625` on average.

This is exactly the failure mode the project wants to avoid: apparent adhesion reduction obtained by over-cutting or fragmenting anatomy.

## Decision

Do not launch R205 learned anatomy-aware refiner training from this feasibility result.

The next route should not be another broad anatomy-aware / keepbone / cutgap / local bridge-edit branch unless it first includes a component-preserving mechanism stronger than simple gap deletion.

Recommended next actions:

1. Complete representative baseline context under R201, especially Swin/U-Net/TransUNet-style baselines already available or feasible.
2. Build hard-case visualizations from R203 to document the exact failure modes.
3. If returning to model improvement, design a component-preserving instance-level model or data/protocol route, not another pixel-deletion refiner.
