# R175 Bridge/Boundary Constraint Plan

**Date**: 2026-07-03

**Current valid best**: R110, clean-test-v2 Dice `0.9177231563529792`.

**Target**: exceed original ARAA by `+0.02`, clean-test-v2 Dice target `> 0.9317660066557425`.

## Decision

The next project push should focus on task-specific bridge and boundary evidence, not another broad source-model sweep.

Current evidence does **not** support claiming that the project already beats SOTA overall. R110 remains the strongest valid local result on Dice, precision, and Boundary IoU among the completed project branches, but the target gap is still open. The useful route is to make the paper claim more exact:

- region accuracy remains the primary clinical segmentation metric;
- bone-gap adhesion is measured as an explicit separation failure;
- boundary quality is measured with boundary-sensitive metrics;
- component topology is measured because epiphysis labels are multi-component masks, not a single blob.

This lets the project show advantage on the failure mode that matters for掌骨骨骺: adjacent bones should stay separated and the epiphyseal contour should stay readable.

## R175 Metric Audit Result

Audit output: `outputs/analysis/r175_bridge_boundary_claim_metrics.json`

The first metrics-only audit used existing clean-test-v2 JSON files. No thresholds were tuned on clean-test-v2.

| System | Dice | Boundary IoU | Precision | Recall | Low-Dice Cases | Low-Boundary Cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R090 patch arbitrator | `0.916440` | `0.251366` | `0.906271` | `0.928080` | `16` | `28` |
| R100 patch complement | `0.917618` | `0.251759` | `0.912686` | `0.924039` | `15` | `29` |
| R110 current best | `0.917723` | `0.251896` | `0.913343` | `0.923580` | `15` | `29` |
| R130 bridge penalty source | `0.890938` | `0.180672` | `0.855765` | `0.931193` | `52` | `47` |
| R140 reviewed variant source | `0.885500` | `0.174439` | `0.854157` | `0.920853` | `63` | `54` |
| R143 high-res medical recipe | `0.891375` | `0.161392` | `0.821758` | `0.975703` | `51` | `58` |
| R165 filtered reannotation source | `0.890303` | `0.184990` | `0.878265` | `0.905050` | `47` | `48` |

Interpretation:

- R110 is still the strongest completed valid system on the standard and boundary metrics that are already available.
- Source-model attempts that explicitly pushed recall or bridge penalties reduced Dice and Boundary IoU on clean-test-v2.
- The current false-bridge metric is not available for all old systems, so it should not be used as a final fairness claim until recomputed from masks for every compared system.
- The remaining opportunity is not “more recall”; it is controlled separation with boundary-preserving edits.

## Claim Metrics To Standardize

### Primary Metrics

1. Dice / IoU: keep for comparability with prior ARAA and segmentation baselines.
2. Boundary IoU: boundary-focused quality, aligned with the Boundary IoU paper.
3. Bone-gap false-positive rate: predicted foreground inside GT-adjacent background bands between epiphysis components.
4. Component-count error and under-segmentation rate: whether adjacent epiphyses are merged.

### Secondary Metrics

1. Boundary F1 or Normalized Surface Dice: more interpretable contour agreement for medical segmentation.
2. Hausdorff95 / surface distance: useful only if pixel spacing or a stable pixel tolerance is documented.
3. clDice/topology score: use cautiously. It is designed for tubular connectivity; for epiphysis, the better adaptation is “separation preservation,” not connectivity preservation.

## Constraint Direction

### Keep

- Candidate-mask local arbitration, because R110 is still the strongest family.
- Boundary-aware evaluation, because low Boundary IoU remains a visible failure signal.
- Component and bridge metrics, because the task is multi-epiphysis separation.
- Train/val-only label protocol review, but only after real reviewed CSV / corrected labels exist.

### Stop

- Simple recall-heavy direct segmenters. R143 shows high recall can worsen bridging.
- Another DINOv3 instance-separation loss sweep. R130/R140/R165 did not become useful sources.
- Clean-test-v2 threshold tuning. It can diagnose but must not choose deployable parameters.
- False SOTA claims from incomplete bridge metrics. Recompute all systems from masks first.

### Replace If Needed

If “骨缝粘连” is too hard to support directly from available binary masks, replace it with these publishable constraints:

- separation-band foreground suppression;
- boundary-band edit Dice;
- under-segmentation component error;
- hard-case subgroup Dice on predeclared bridge/boundary cases selected from train/val or locked before evaluation.

## Next Runs

### R176: Full Mask-Synced Constraint Audit

Purpose: compute the same bridge/boundary/topology metrics for R090/R100/R110 and selected SOTA-like baselines from actual prediction masks.

Requirements:

- sync or regenerate prediction masks for R090, R100, R110, R130/R143/R165;
- run `scripts/audit_r175_bridge_boundary_claim_metrics.py` with `NAME=METRICS_JSON:PRED_DIR`;
- produce a final table with Dice, Boundary IoU, separation-band FP rate, component-count error, and under-segmentation rate.

Decision gate:

- If R110 already dominates bridge/boundary metrics, write the advantage claim around evaluation evidence.
- If R110 does not dominate bridge metrics, do not claim SOTA superiority; launch R177 as a targeted constraint model.

### R177: Boundary-Preserving Separation Arbitrator

Purpose: improve only the bridge/boundary failure region while preserving R110 elsewhere.

Mechanism:

- start from R110 as immutable anchor;
- restrict edits to predicted/label-derived separation-risk bands on train/val;
- add a small local arbitrator trained to remove foreground in likely inter-epiphysis gaps and recover boundary-adjacent false negatives;
- validate only on original val or a train/val-derived hard-case validation subset;
- run clean-test-v2 once after protocol is frozen.

Success criteria:

- Dice beats R110, not just source-model baselines;
- Boundary IoU improves over R110;
- separation-band FP rate and under-segmentation rate decrease without recall collapse.

### R178: Literature-Backed External Route Only If Environment Changes

Purpose: reopen Mask DINO / Mask2Former-style architecture only with a CUDA-toolkit-capable isolated environment.

Gate:

- `nvcc`, Detectron2 dependencies, and custom-op smoke must pass outside `yolo-sam-gpu`;
- 1-2 image overfit must produce useful non-empty masks before any full run.

## Literature Anchors

- **Boundary IoU: Improving Object-Centric Image Segmentation Evaluation**. Use this to justify Boundary IoU / Boundary AP style evaluation.
- **Family of boundary overlap metrics for the evaluation of medical image segmentation**. Use this to justify medical boundary overlap metrics.
- **A Generalized Surface Loss for Reducing the Hausdorff Distance in Medical Imaging Segmentation**. Use this to justify surface-distance losses or Hausdorff-aware reporting.
- **clDice: A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation**. Useful as a topology-loss reference, but adapt carefully because our task needs separation, not tubular connectivity.
- **Mask DINO: Towards a Unified Transformer-based Framework for Object Detection and Segmentation**. Keep as the external instance-segmentation architecture reference if the environment allows faithful implementation.

## Immediate Next Action

Do R176 before any new GPU training. The project needs synchronized prediction masks for the current best and core baselines so bridge/boundary claims are computed fairly. If masks are unavailable locally, regenerate or sync them from the server, one experiment at a time.
