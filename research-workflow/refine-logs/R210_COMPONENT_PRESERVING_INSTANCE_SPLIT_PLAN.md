# R210 Component-Preserving Instance Split Plan

Date: 2026-07-06

## Goal

Build the next model-changing experiment after R209-F0. R210 should reduce bone-gap adhesion and adjacent-bone merge errors while preserving R110's region quality. This is not another broad anatomy-aware run and not a free pixel-deletion refiner.

## Evidence From Previous Runs

- R110 is strong on clean-test-v2 overlap/boundary compared with ARAA, but it is not strong on component diagnostics.
- R177 proves free local edits can remove bridges but fragments anatomy badly.
- R179 proves instance-separation supervision can improve some topology metrics but direct segmentation sacrifices too much Dice and boundary quality.
- R184/R186/R204 prove guarded morphology often collapses to no-op.
- R205-F0 proves gap-FP deletion can create misleading metric gains while worsening component fragmentation.
- R209-F0 proves a component-preserving route is feasible on val:
  - `72/96` val images have predicted multi-instance merge.
  - anchor component merge rate: `0.343750`.
  - anchor component count MAE: `3.020833`.
  - anchor gap FP: `0.188199`.

## Core Hypothesis

The current failure is better framed as component/instance separation inside R110 masks than as generic boundary smoothing or gap-pixel deletion. A valid model should split merged predicted components only when the split preserves expected instance/component structure.

## Dataset And Split Rules

- Dataset: `TSRS_RSNA-Epiphysis`.
- Training: original `train`.
- Model selection: original `val`.
- Anchor: true R110 train/val masks from `r110_r100_r108_patch_basic_trainval`.
- Final apply anchor: R110 clean-test-v2 masks from `r110_r100_r108_patch_basic`.
- clean-test-v2 remains locked until the val gate passes.
- No `TSRS_RSNA-Articular-Surface`.
- No raw dataset modification.

## Proposed Method

R210 should use an anchor-conditioned instance split model:

1. Inputs:
   - grayscale radiograph,
   - R110 anchor mask,
   - signed distance to anchor boundary,
   - local image gradient,
   - local anchor density,
   - candidate gap/separation band features.
2. Targets:
   - GT binary mask,
   - GT instance separation band,
   - optional center/hover-style vectors or instance-boundary targets from existing instance labels.
3. Output:
   - split likelihood inside merged R110 components,
   - optional refined foreground confidence only for acceptance scoring, not free deletion.
4. Application:
   - identify R110 connected components likely covering multiple GT-like instances,
   - split only within high-confidence internal separation bands,
   - keep foreground mass unless a component-safe rule accepts the change.

## Implementation Preference

Prefer a new script rather than modifying old failed scripts in place:

- `scripts/train_r210_component_preserving_instance_split.py`

Reuse code patterns from:

- `scripts/train_r177_separation_arbitrator.py` for R110-anchor feature extraction and train/val/apply split handling.
- `scripts/train_timm_instance_separation_segmenter.py` / `scripts/train_instance_separation_segmenter.py` for instance separation targets.
- `scripts/run_r201_unified_eval.py` for final metric reporting.
- `scripts/run_r209_component_preserving_feasibility_audit.py` for component/instance merge diagnostics.

## Validation Gate

R210 may evaluate clean-test-v2 only if the selected val configuration satisfies all of:

- nonzero accepted edits on R209 merge-heavy val cases;
- mean Dice delta vs R110 train/val anchor >= `-0.003`;
- mean IoU delta vs anchor >= `-0.003`;
- mean Recall delta vs anchor >= `-0.005`;
- Boundary IoU or Boundary F1 improves;
- gap-region FP rate decreases;
- component merge rate decreases or component count MAE decreases;
- component count MAE must not increase;
- hard-case visualization shows no over-erosion, no missing epiphyses, and no obvious fragmentation.

If these are not met, R210 is a no-go and must not touch clean-test-v2.

## Hard-Case Validation Set

Use the R209 top merge-heavy val cases for qualitative gate:

- `3700.png`
- `3460.png`
- `1886.png`
- `1931.png`
- `1742.png`
- `3725.png`
- `1880.png`
- `1901.png`
- `1452.png`
- `13867.png`
- `2044.png`
- `3840.png`

Each visualization should show image, GT instance mask, R110 anchor, R210 output, split map, and accepted/rejected split regions.

## Expected Deliverables

- R210 training/apply script.
- R210 validation summary JSON/CSV.
- R210 hard-case visualization index.
- If val gate passes:
  - R210 clean-test-v2 predictions under an isolated output exp.
  - R201 unified metrics JSON/CSV for R210.
  - comparison against ARAA and R110.
- If val gate fails:
  - no clean-test-v2 application,
  - no-go report explaining which gate failed.

## Stop Conditions

Stop R210 and pivot if:

- accepted edits are near-zero like R184/R186/R204;
- component count MAE increases like R177/R205-F0;
- Dice/IoU/Recall drop exceeds gate tolerance;
- visualizations show reduced bridges are caused by erosion or missing structures.

Decision: implement R210 next as a component-preserving instance split model with strict validation gating.
