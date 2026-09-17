# R205 Anatomy-Aware Prior Attempt Audit

Date: 2026-07-06

## Question

Has anatomy-aware segmentation already been tried in this project?

Answer: yes. The broad idea has been tried repeatedly. A new run should not be justified by the phrase "anatomy-aware" alone. It must be materially different from the prior failed or saturated branches.

## Prior Branches

### 1. `allstack_anatomy_roi_*` / `keepbone_cutgap`

This was an older main internal family, including keep-bone / cut-gap / AIC / uncertainty / boundary-prefgate variants.

Observed pattern:

- It produced strong anchors before R110.
- It plateaued around the low `0.91` Dice range on clean-test-v2.
- It did not establish a decisive bone-gap adhesion or component-consistency improvement.

Decision:

- Do not relaunch this family unchanged.
- Do not rename it as R205.

### 2. R177 Boundary-Preserving Separation Arbitrator

Purpose:

- Local bridge/boundary arbitrator selected on train/val, applied once to clean-test-v2.

Outcome:

- Clean-test-v2 Dice: `0.900540`.
- Boundary IoU improved to `0.265109`.
- False-bridge reduced to `0.0`.
- Component-count error exploded to `143.901235`.

Interpretation:

- It can remove bridges, but the free pixel-edit policy fragments masks badly.

Decision:

- Do not allow free add/delete edits without component-level acceptance.

### 3. R179 Boundary/Gap Constrained DINOv3 Instance-Separation Training

Purpose:

- Move bridge/gap constraints into training-time loss.

Outcome:

- Clean-test-v2 Dice: `0.892977`.
- Boundary IoU: `0.187623`.
- Component-count error improved to `0.938272`.
- False-bridge improved to `0.296296`.

Interpretation:

- Training-time topology constraints can improve structural metrics, but this branch sacrifices too much region and boundary quality.
- R180 found only `5/81` images where R179 beat R110; GT-leaking oracle reached only `0.918213`.

Decision:

- Do not run another broad loss-weight sweep of the same DINOv3 instance-separation route.

### 4. R184 True-R110 Guarded Arbitrator

Purpose:

- Repeat the local repair idea using true R110 train/val masks instead of the earlier proxy anchor.

Outcome:

- Validation gate technically passed only as a no-op.
- Clean-test-v2 exactly matched R110.
- Edited fraction: `0.0`.

Decision:

- Do not accept a gate that passes with zero or near-zero edits.

### 5. R186 Fast Neck/Layout Gate

Purpose:

- Replace slow morphology scanning with a faster neck candidate rule.

Outcome:

- Validation nonzero edit existed, but edit fraction was only `2.025825e-07`.
- Clean-test-v2 Dice delta vs R110: `-0.00000295`.
- Component-count error improved, but edit magnitude was too tiny.

Decision:

- Simple geometry-only postprocessing is not enough.

### 6. R204 Conservative Morphology Gate

Purpose:

- Conservative bridge/gap edit gate after R203 failure manifest.

Outcome:

- 24-image val smoke: `val_gate_failed`.
- All 16 tested configs had `edited_frac=0`.
- No clean-test-v2 application.

Decision:

- Do not expand this morphology grid further unless candidate generation is redesigned.

## What This Means for R205

R205 should not be "another anatomy-aware refiner" in the old sense.

R205 is only justified if it satisfies all of the following:

- Uses true R110 train/val anchor masks.
- Has an explicit nonzero-edit validation requirement.
- Preserves Dice/IoU/Recall relative to R110 on val within strict tolerance.
- Separates four mechanisms in the log:
  - boundary/distance supervision,
  - gap-exclusion penalty,
  - recall-preservation penalty,
  - component acceptance or component diagnostics.
- Uses validation-only threshold/checkpoint selection.
- Applies clean-test-v2 only once after val gate passes.
- Produces hard-case visualizations showing reduced adhesion without over-erosion.

If these cannot be implemented cleanly, skip R205 and move to a different route:

- representative baseline completion under R201, such as Swin/U-Net/TransUNet;
- data/protocol route after reviewed train/val labels are available;
- a genuinely new model family with strong instance separation, not another local morphology or loss-weight sweep.

## Updated Recommendation

Do not immediately launch R205 training.

First implement a small R205 feasibility audit:

1. On train/val only, build a candidate edit map from R110 error regions.
2. Estimate whether there are enough safe, nonzero corrections that improve gap/component metrics without hurting recall.
3. If safe corrections are sparse like R186/R204, stop.
4. If safe corrections exist, then train the learned correction head.

This keeps the project from repeating the old anatomy-aware loop under a new name.
