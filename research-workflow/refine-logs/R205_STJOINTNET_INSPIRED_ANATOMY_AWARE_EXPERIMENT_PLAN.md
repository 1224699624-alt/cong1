# R205 ST-JointNet-Inspired Anatomy-Aware Experiment Plan

Date: 2026-07-06

## Purpose

Prepare the next YOLO+SAM epiphysis improvement experiments using the key lesson from Rezaei et al., *Annotation-Free Anatomy-Aware Joint Segmentation in Pediatric Hand Radiographs for Explainable Bone Age Estimation*: do not rely on blind morphology cuts. Use anatomy-aware supervision, distance/boundary cues, and explicit bone/gap exclusion so that adhesion is reduced without over-erosion or missed epiphysis regions.

This is a project-improvement plan, not paper writing. The target is to improve boundary and anatomy-consistency metrics under the frozen R201 protocol while keeping Dice/IoU competitive with ARAA/R110.

## Fixed Constraints

- Task: `TSRS_RSNA-Epiphysis` only.
- Protocol: R201 unified evaluation.
- Final diagnostic/evaluation split: `TSRS_RSNA-Epiphysis_clean_test_v2/test`.
- clean-test-v2 must not be used for threshold search, hyperparameter tuning, or model selection.
- All tuning and selection must use original train/val only.
- Raw dataset structure must not be modified.
- Outputs must be isolated under new R205/R206 folders.

## Current Evidence

- ARAA and R110 are restored in R201 with `num_evaluated=81`.
- R110 improves Dice/IoU/boundary over ARAA but worsens component merge and component count error.
- R202 nnU-Net is strong but treated as an internal risk-audit baseline, not the target to beat on Dice/IoU.
- R204 conservative morphology gate failed on a 24-image val smoke: `val_gate_failed`, `clean_test_applied=false`, and all tested configs had `edited_frac=0`.
- Conclusion: the next method should move from hard-coded bridge cutting to learned anatomy-aware correction.

## Prior Anatomy-Aware Attempts and What R205 Must Not Repeat

The project has already tried several anatomy/gap-aware families. R205 is only worth launching if it is materially different from these failed or saturated routes.

- Older `allstack_anatomy_roi_*` and `keepbone_cutgap` refiner family:
  - This was the previous anatomy-aware line and produced strong internal anchors before R110.
  - It plateaued around Dice `0.91` on clean-test-v2 and did not provide a decisive bridge/gap claim.
  - Do not relaunch this family as-is.
- R177 boundary-preserving separation arbitrator:
  - It improved boundary IoU but fragmented masks badly.
  - Clean-test-v2 Dice dropped to `0.900540`; component-count error exploded.
  - Do not allow free pixel add/delete around bridge-risk regions without strict component preservation.
- R179 boundary/gap constrained DINOv3 instance-separation training:
  - It moved bridge/gap constraints into training and improved topology-style metrics.
  - It dropped clean-test-v2 Dice to `0.892977` and Boundary IoU to `0.187623`.
  - Do not run another broad loss-weight sweep of this same source-model route.
- R184 true-R110 guarded local arbitrator:
  - It used true R110 train/val masks, but the selected guarded model became a no-op.
  - Clean-test-v2 matched R110 exactly.
  - Do not accept a validation gate that passes with near-zero edits.
- R186 fast neck/layout gate:
  - It produced tiny nonzero edits and slightly improved structural diagnostics.
  - Clean-test-v2 Dice/Boundary IoU were effectively unchanged, and edit fraction was only `7.66e-08`.
  - Do not rely on geometry-only postprocessing to close the current gap.
- R204 conservative morphology gate:
  - The 24-image val smoke failed and all configs had `edited_frac=0`.
  - Do not expand this morphology grid further unless candidate generation is redesigned.

Therefore R205 must be framed as a stricter delta over prior work:

- true R110 anchor only, not old R025b/R038 anchors and not a proxy anchor;
- train/val-only selection with an explicit nonzero-edit requirement;
- distance-transform or boundary regression supervision, not only binary mask loss;
- gap-exclusion and recall-preservation losses reported separately;
- val gate must show both anatomy-metric improvement and no recall/Dice collapse before clean-test-v2 application;
- if these requirements cannot be implemented, skip R205 and move to a different model family or data-quality route.

## Paper-Derived Design Lessons

From ST-JointNet:

- Use distance-transform supervision to improve thin boundary localization.
- Use anatomy-aware exclusion loss so predictions are penalized in forbidden bone/gap regions.
- Use coarse-to-fine correction instead of one-shot morphology.
- Use expected component structure as a consistency check, but do not force it by aggressive erosion.

For our task, this becomes:

- The anchor mask is R110.
- The correction model edits only local uncertain/boundary/gap candidate regions.
- Training uses train/val GT only.
- Selection gate rejects any model that improves gap/component metrics by sacrificing recall, Dice, or boundary quality.

## R205 Feasibility Gate: Do Not Train Until This Passes

Because broad anatomy-aware variants have already been tried, R205 must begin with a feasibility audit rather than another GPU training run.

R205-F0 must answer:

- Are there enough train/val R110 error regions where a local correction can safely reduce gap/component errors?
- Do safe corrections require additions, deletions, or both?
- Are candidate edits larger than the near-zero edit regime seen in R184/R186/R204?
- Can any candidate rule improve validation anatomy diagnostics without dropping Dice/Recall?

Required output before training:

- `outputs/analysis/r205_feasibility_safe_edit_audit.json`
- `outputs/analysis/r205_feasibility_safe_edit_audit.csv`
- A short decision entry in `STEP_SYNC_LOG.md`

Go condition:

- nonzero safe edits on validation;
- mean edited fraction clearly above the R186 near-zero regime;
- validation Dice/Recall not worse than R110 beyond strict tolerance;
- at least one of gap FP, component merge, or component count MAE improves on validation.

No-go condition:

- safe edits are sparse/no-op;
- improvements only come from deleting foreground and reducing recall;
- the audit reproduces R177 fragmentation risk or R184/R204 no-op behavior.

Only if R205-F0 passes should the learned refiner below be launched.

## R205 Main Experiment: Anatomy-Aware Boundary/Gap Refiner

### Claim Tested

An anatomy-aware learned refiner can reduce bone-gap adhesion and component merge errors relative to R110 while preserving Dice/IoU and improving boundary metrics.

### Implementation Anchor

Start from existing project code rather than a new framework:

- Preferred base script: `scripts/train_boundary_band_refiner.py`
- Alternative base if more channels/loss hooks are needed: `scripts/train_error_refiner.py`
- Anchor masks:
  - train/val: `outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis/{train,val}/masks`
  - clean-test-v2 application only after val gate: `outputs/ablations_variants/r110_r100_r108_patch_basic/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks`

### Inputs

Use a compact feature stack:

- grayscale image
- R110 anchor mask
- R110 boundary map
- signed distance transform of R110
- local image gradient
- train/val-only GT-derived gap/boundary target maps during training

Do not provide clean-test-v2 GT-derived maps at inference.

### Loss Terms

Minimal required loss:

- segmentation loss: BCE/Dice against GT
- boundary/distance loss: distance-transform or boundary-band loss
- gap-exclusion loss: penalize predicted foreground in train/val GT gap region
- recall-preservation loss: penalize false negatives inside GT foreground, especially away from boundary

Optional if stable:

- component proxy penalty on small validation batches
- clDice/topology-style regularizer, only if it does not reduce recall

### Validation Gate

Select model and threshold on val only. A candidate passes only if:

- Dice >= R110 val Dice - 0.001
- IoU >= R110 val IoU - 0.001
- Recall >= R110 val Recall - 0.002
- Boundary IoU and Boundary F1 are higher than R110 val
- gap FP is lower than R110 val
- component merge rate is lower than R110 val, or at least not worse while component count MAE improves
- hard-case visual spot check shows no obvious over-erosion or missed epiphysis

Only after this gate passes may the locked checkpoint and locked threshold be applied once to clean-test-v2.

### Clean-Test Evaluation

If val gate passes:

1. Apply the locked R205 model to clean-test-v2 R110 masks.
2. Save masks under:
   `outputs/ablations_variants/r205_anatomy_aware_gap_refiner/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks`
3. Register or evaluate with R201.
4. Compare against ARAA, R110, R202 risk audit, and Swin/U-Net baselines where available.

Success on clean-test-v2 requires:

- Dice/IoU not meaningfully below R110 and above or comparable to ARAA.
- Boundary IoU/F1, Surface Dice, HD95, ASSD better than R110.
- gap FP, component merge, and component count MAE better than R110.
- Qualitative hard cases show less adhesion without over-cutting.

## R205A Smoke Run

Purpose: verify that the model can learn nonzero conservative corrections on val without collapsing recall. This run is allowed only after R205-F0 passes.

Suggested remote command shape:

```bash
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
conda activate yolo-sam-gpu
screen -dmS r205_smoke_gap_refiner bash -lc '
python scripts/train_boundary_band_refiner.py \
  --anchor-exp r110_r100_r108_patch_basic_trainval \
  --apply-anchor-exp r110_r100_r108_patch_basic \
  --output-exp r205_smoke_anatomy_aware_gap_refiner \
  --checkpoint outputs/context_segmenter/r205_smoke_anatomy_aware_gap_refiner/best.pt \
  --history-json outputs/context_segmenter/r205_smoke_anatomy_aware_gap_refiner/history.json \
  --metrics-json outputs/analysis/r205_smoke_anatomy_aware_gap_refiner_clean_test_v2_metrics.json \
  --control-metrics-json outputs/analysis/r205_smoke_anatomy_aware_gap_refiner_original_test_metrics.json \
  --epochs 12 --min-epochs 4 --patience 4 \
  --limit-train 96 --limit-val 32 \
  --batch-size 4 --img-size 512
'
```

Important: if the current script cannot express gap-exclusion/distance losses cleanly, do not treat this as final R205. Use it only as a pipeline smoke, then patch the script.

## R205B Full Val-Gated Run

Purpose: train the actual anatomy-aware gap refiner after the loss hooks are implemented.

Required before launch:

- add train/val-only gap-exclusion map construction
- add distance-transform auxiliary target
- add validation summary with R110 anchor comparison
- write locked threshold/checkpoint to result JSON
- prevent clean-test-v2 application unless val gate passes

Outputs:

- `outputs/context_segmenter/r205_anatomy_aware_gap_refiner/best.pt`
- `outputs/analysis/r205_anatomy_aware_gap_refiner_val_summary.json`
- `outputs/analysis/r205_anatomy_aware_gap_refiner_result_summary.json`
- optional clean-test metrics only after val gate:
  `outputs/analysis/r205_anatomy_aware_gap_refiner_clean_test_v2_metrics.json`

## R206 Baseline Completion

In parallel or after R205 smoke, prepare representative non-nnU-Net baselines:

- Swin-UNet / SwinUNETR if feasible in current environment.
- TransUNet or strong U-Net if SwinUNETR setup becomes slow.
- Existing R106 Swin tiny U-Net and R143 high-res U-Net should be re-evaluated under R201 if masks are complete.

These are comparison baselines, not the main project improvement route.

## Run Order

1. Run R205-F0 train/val feasibility audit before any GPU training.
2. If R205-F0 fails, stop R205 and do not relaunch old anatomy-aware routes.
3. If R205-F0 passes, finish R205 loss/design patch on top of the existing refiner script.
4. Run R205A smoke on train/val only.
5. If smoke learns nonzero edits without recall collapse, run R205B full train/val.
6. If R205B val gate passes, apply once to clean-test-v2 and run R201.
7. Generate hard-case visualizations for R203 clean bridge candidates and recall-risk cases.
8. Re-evaluate Swin/U-Net style baselines under R201 for context.

## Stop Conditions

Stop R205 and redesign if:

- val Dice drops by more than 0.001 without large anatomy-metric gain
- recall drops by more than 0.002
- gap/component improvements come mainly from foreground erosion
- hard-case visualizations show broken epiphysis regions
- edits remain near zero, repeating R204 failure

## Expected Next Decision

If R205 succeeds on val, it becomes the next clean-test candidate.
If it fails because edits are too weak, move to a stronger coarse-to-fine correction head.
If it fails because edits over-cut, increase recall-preservation and restrict edits to learned high-confidence adhesion zones.
