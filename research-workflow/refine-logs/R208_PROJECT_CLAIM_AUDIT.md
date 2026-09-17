# R208 Project Claim Audit

Date: 2026-07-06

Verdict type: local result-to-claim judgment, pending external review.

## Intended Project Claim

The project aims to improve YOLO+SAM epiphysis segmentation by keeping Dice/IoU competitive with ARAA/R110 while clearly improving bone-gap adhesion, adjacent-bone misconnection, boundary quality, and anatomical consistency diagnostics.

## Evidence Used

- R201 unified evaluation table: `outputs/analysis/r201_unified_eval/r201_unified_comparison_table.csv`
- R205-F0 feasibility audit: `outputs/analysis/r205_feasibility_safe_edit_audit_val.json`
- R206 hard-case visualizations: `outputs/analysis/r206_hardcase_visualizations/index.html`
- R207 baseline coverage audit: `outputs/analysis/r207_baseline_coverage_audit.json`

## Key Numbers

### ARAA vs R110

| Metric | ARAA | R110 | R110 Direction |
|---|---:|---:|---|
| Dice | 0.911766 | 0.917723 | better |
| IoU | 0.838436 | 0.848570 | better |
| Boundary IoU | 0.235265 | 0.251896 | better |
| Boundary F1 | 0.374350 | 0.394721 | better |
| Surface Dice 2px | 0.677752 | 0.684156 | better |
| HD95 px | 6.596897 | 7.220072 | worse |
| ASSD px | 2.343013 | 2.419143 | worse |
| gap-region FP rate | 0.201259 | 0.201358 | tied/slightly worse |
| component merge rate | 0.679012 | 0.790123 | worse |
| component count MAE | 2.086420 | 2.506173 | worse |

### R202 nnU-Net Risk-Audit Baseline

R202 has the strongest standard and boundary metrics:

- Dice: `0.923573`
- IoU: `0.858508`
- Boundary IoU: `0.270431`
- Boundary F1: `0.417773`
- Surface Dice 2px: `0.713582`
- HD95: `6.113146 px`
- ASSD: `2.083621 px`

But it does not solve every anatomy diagnostic:

- gap-region FP rate: `0.218767`, worse than both ARAA and R110.
- component merge rate: `0.679012`, tied with ARAA and better than R110.
- component count MAE: `1.827160`, better than ARAA/R110.

### R205-F0 Safe-Edit Audit

The GT-guided gap-FP deletion upper bound improved many validation metrics:

- Dice delta: `+0.036974`
- Boundary IoU delta: `+0.241213`
- gap-region FP delta: `-0.188199`
- Recall delta: `0.000000`

But it fragmented anatomy:

- component count MAE delta: `+22.562500`
- safe images: `15/96`
- mean component safe: `false`
- decision: `no_go_do_not_train_r205`

This invalidates another simple anatomy-aware/pixel-deletion refiner route.

## Claim Judgment

claim_supported: partial

confidence: medium

## What The Results Support

The results support a limited claim:

- R110 is a strong internal anchor that improves Dice, IoU, Boundary IoU, Boundary F1, and Surface Dice 2px relative to ARAA.
- R201 is now a credible unified evaluation protocol with overlap, boundary/surface, and failure-mode diagnostic metrics.
- Bone-gap/component diagnostics expose real failure modes that Dice/IoU and Boundary IoU alone hide.
- R205-F0 confirms that simply deleting gap false positives can create misleading gains while fragmenting components; this justifies rejecting pixel-deletion refiners.
- R206 provides qualitative hard-case panels for ARAA/R110/R202 to inspect adhesion, boundary, and fragmentation risks.

## What The Results Do Not Support

The current evidence does not support the main desired project claim yet:

- It does not show that the current YOLO+SAM method reduces bone-gap adhesion relative to ARAA.
- It does not show that R110 improves component merge or component count MAE relative to ARAA.
- It does not show that a new method improves anatomy consistency while preserving Dice/IoU.
- It does not support another broad anatomy-aware/keepbone/cutgap training run.
- It does not support claiming that better boundary metrics automatically mean fewer anatomical misconnections.

## Missing Evidence

To support the intended project claim, the project still needs a new candidate that passes all of the following:

- Dice and IoU at least comparable to ARAA/R110.
- Boundary IoU/F1 and surface metrics at least comparable to R110, ideally better.
- gap-region FP rate better than ARAA/R110.
- component merge rate and component count MAE better than R110 and preferably ARAA.
- R206-style hard-case visualizations showing reduced adhesion without fragmentation, over-erosion, or missed epiphysis regions.

## Route Decision

Do not continue:

- anatomy-aware pixel-deletion refiners,
- morphology/neck-cut grids,
- R179-style broad loss-weight sweeps,
- R184/R204-style conservative gates that become no-op.

Next viable routes:

1. Component-preserving instance-level model:
   - Treat adjacent epiphysis structures as separable instances or ordered components before unioning to binary.
   - Candidate families: instance separation with explicit component count/assignment, Mask2Former/MaskDINO-style binary-union only if training is stable, or a lightweight component-preserving readout over R110/R202 features.
2. Data/protocol route:
   - Revisit reviewed train/val label protocol if R168/R170 artifacts become available.
   - Use it to address label ambiguity around true growth plates vs false bridges.
3. Baseline completion only if needed:
   - R201-ready baselines already include ARAA, R110, R143, and R202.
   - R106/R097 backfill is optional and should not block model improvement.

## Suggested Working Claim For Now

Current defensible claim:

> R110 improves overlap and several boundary metrics over ARAA, but the unified R201 diagnostics reveal unresolved anatomical consistency failures, especially component merge/count errors. Simple gap-deletion and morphology-based repair routes are unsafe because they either fragment anatomy or become no-op.

Current not-yet-supported claim:

> The proposed method solves bone-gap adhesion and anatomical consistency while preserving Dice/IoU.

## Immediate Next Experiment Recommendation

Before another training run, design R209 as a component-preserving instance-level feasibility plan:

- Use train/val only.
- Estimate whether component assignment/order information can separate R110 merged structures without deleting foreground.
- Require component count MAE and merge rate to improve without Dice/Recall collapse.
- Only then consider training a new model.
