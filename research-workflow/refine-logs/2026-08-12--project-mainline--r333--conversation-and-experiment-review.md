---
type: results-report
date: 2026-08-12
experiment_line: dual-interaction-prior
round: 333
purpose: conversation-and-experiment-review
status: active
source_artifacts:
  - research-workflow/refine-logs/R260_MATURE_NNUNET_PRIOR_SPEC.md
  - outputs/analysis/r317_image_centers_only_checkpoint_selection.json
  - research-workflow/refine-logs/R320_PAPER_READY_EXPERIMENT_TABLES.md
  - research-workflow/refine-logs/R332_RAM_LOCKED_TEST_VS_R325_RESULT.md
  - research-workflow/refine-logs/R333_DUAL_INTERACTION_PRIOR_EXPERIMENT_PLAN.md
linked_experiments:
  - TSRS seam-prior line
  - RAM overlap-completion line
linked_results:
  - R317 TSRS original-val
  - R332 RAM locked test
---

# Dual-Interaction Prior / Round 333 / Conversation and Experiment Review / 2026-08-12

## 1. Executive Summary

The project evolved from direct YOLO+SAM mask cutting into a prior-guided, loss-constrained and backbone-transferable segmentation framework. Two mechanisms are now supported separately:

1. TSRS: an input-side continuous seam/background probability prior plus separation loss can improve Dice/IoU slightly and improve boundary/surface and merge diagnostics. R317 is the current valid TSRS development result.
2. RAM-W600: an output-side iterative instance-completion prior can improve a strong native-resolution nnU-Net on locked test. R332 is the current strongest RAM result.

The first direct combination, R333, did not pass. RAM lost part of R332's overlap performance after backbone unfreezing; TSRS used a mismatched 512-pixel adapter instead of the mature R317 nnU-Net preprocessing path. This is an implementation/training-alignment failure, not evidence that the conditional unified framework is impossible.

## 2. Conversation and Decision Evolution

### Phase A: direct post-processing and reverse-topology exploration

- Initial target: reduce adjacent-bone bridges and poor boundaries without materially sacrificing Dice/IoU.
- The key conceptual correction was that TSRS should preserve background seams, not encourage foreground connectivity.
- R244 oracle proved that correctly selected local cuts can improve Boundary IoU/F1, gap FP and component count without Recall loss.
- R240-R248 repeatedly showed that GT-free scalar/crop selection could not distinguish a true seam cut from foreground erosion safely. The bottleneck was candidate safety, not absence of useful cuts.

Decision: stop relying on test-time cut selection and learn a prior/supervision signal during training.

### Phase B: developmental and relation priors

- Age, sex, X-ray appearance, centers and local scale were explored as conditions for a probabilistic relation/seam heatmap.
- R256 showed development-conditioned signal but also redundant explicit geometry.
- R257/R257b established that image-only center localization was feasible and repaired the unstable scale head.
- R258 provided patient-disjoint OOF prediction-derived centers/scales.
- R258B found genuine age/sex signal, but it increased false separation and failed specificity/additional-value gates.
- After user review, age/sex was removed from the principal method because it exists only for TSRS and did not support a cross-dataset framework.

Decision: retain image/context/center-derived spatial priors; do not use age/sex as a universal condition.

### Phase C: mature TSRS seam-prior segmentation

- R259 showed that an immature nnU-Net probe was not interpretable.
- R260 restarted from a mature R202 nnU-Net, added a zero-initialized prior input channel and a background-separation loss. It produced the first convincing mature-prior result.
- R265-R272 explored anatomical relations, ROI corrections, absolute seam loss and instance-derived seam supervision. They confirmed learnability but also exposed over-suppression, ROI mis-targeting and competition between seam supervision and foreground segmentation.
- R279-R283 tested fixed support, lower support, gradient normalization and spatial/confidence-gated support. These did not beat the mature baseline consistently; most traded foreground Recall/Dice for ASSD or gap behavior.
- R307 confirmed that simply replacing binary labels with imperfect indexed-instance supervision was not automatically beneficial.
- R317 removed age/sex and retained image/center prior with continuous alpha=0.035. R201 checkpoint selection yielded the current TSRS development anchor.
- R320 organized paired U-Net and nnU-Net results. Both backbones improved in the principal metrics, but results remain single-seed and TSRS original-val only.

Decision: TSRS mainline is continuous input-side seam prior plus local separation loss, with R317 as current anchor. Do not return to global support-weight sweeps.

### Phase D: RAM-W600 and the difference between seam and projection overlap

- RAM labels are 14 independent instance channels and can represent true projected overlap. TSRS background seams and RAM multi-instance overlap have opposite local targets.
- R284b showed a large positive overlap-prior effect on a compact U-Net.
- R285 showed the same simple single-map residual was ineffective on a stronger nnU-Net.
- R286-R288 moved from a global map toward pair-specific local-interface and boundary priors; gains existed but were small.
- R324-R325 established a native-resolution nnU-Net baseline and calibrated overlap loss. Increasing lambda beyond the balanced region produced trade-offs rather than larger useful gains.
- R326 borrowed layer-separation ideas; it slightly beat the frozen baseline but not a capacity-matched generic adapter on key overlap metrics.
- R327-R329 reframed the problem as instance completion from baseline-shaped errors. R329 demonstrated that the shared per-instance prior beat both the baseline and a generic adapter.
- R330 added moderate surface and volume constraints, improving RAM official metrics.
- R331 showed the R330 trend was stable over five seeds.
- R332 jointly trained a multi-step refiner and locked two correction steps on validation, then evaluated the untouched test set once. It is the strongest valid RAM result.

Decision: RAM mainline is output-side, per-instance, iterative overlap completion—not an input-side background seam map.

### Phase E: attempted unified dual-interaction framework

R333 defined three interaction states:

- separation: confirmed background seam;
- overlap: confirmed multi-instance foreground;
- uncertain: no directional interaction loss.

The unified objective was:

`L = Lseg + lambda_sep Lseam + lambda_comp Lcompletion + lambda_cons Lnoninterference`.

The direct implementation failed its validation gates:

- RAM: unfreezing the decoder caused representation drift and weakened R332 overlap performance.
- TSRS: the adapter resized inputs to 512 and used coarse mask remapping instead of R317's native nnU-Net preprocessing/prediction space, so it was not a valid matched R317 integration.

Decision: keep the unified interaction-state formulation, but restore dataset-correct adapters and protect mature backbones.

## 3. Data and Evaluation Protocol

### TSRS_RSNA-Epiphysis

- Primary development split: 875 train / 96 original-val unless an isolated cleaning manifest is explicitly named.
- Manual-review variant: 814 train / 94 val; this is a keep-list, not a replacement for the original dataset.
- TSRS color IDs contain annotation collisions and cannot be interpreted as reliable true overlap labels.
- Current locked rule: clean-test-v2 is final evaluation/diagnosis only and cannot select thresholds, epochs, weights or variants.
- Articular-Surface is always separate and excluded from this line.
- Evaluation: R201, including Dice, foreground IoU, Boundary IoU/F1, Surface Dice 2/5 px, HD95, ASSD; gap FP, merge rate and component-count MAE are task diagnostics.

### RAM-W600

- Official split used in recent records: 425 train / 69 val / 124 test.
- Labels: 14 independent bone-instance channels; overlapping pixels may belong to multiple instances.
- Validation selects checkpoint, correction depth and coefficients; test is locked final evaluation.
- Primary official-style metrics: DSC, NSD@2px, VOE, symmetric MSD, MSD fail rate and RAVD, reported overall, overlap-region and pair-intersection levels.

## 4. Canonical Results

### TSRS: R317 versus mature baseline, original-val

| Metric | Baseline | R317 | Delta | Direction |
|---|---:|---:|---:|---|
| Dice | 0.897112 | 0.902858 | +0.005746 | better |
| IoU | 0.819215 | 0.826369 | +0.007154 | better |
| Boundary IoU | 0.239893 | 0.245245 | +0.005352 | better |
| Boundary F1 | 0.379017 | 0.386746 | +0.007729 | better |
| Surface Dice 2 px | 0.654511 | 0.668408 | +0.013897 | better |
| Surface Dice 5 px | 0.896460 | 0.913312 | +0.016852 | better |
| HD95 px | 54.3414 | 14.0084 | -40.3330 | better |
| ASSD px | 14.1738 | 4.0475 | -10.1263 | better |
| Gap-region FP | 0.205547 | 0.181737 | -0.023810 | better |
| Merge rate | 0.572917 | 0.500000 | -0.072917 | better |
| Component-count MAE | 5.67 | 1.854 | -3.816 | better |

Important caveat: the mature-baseline HD95/ASSD means are dominated by severe outliers, so the large reductions should be accompanied by per-case distributions and qualitative cases. R317 is single-seed development evidence, not yet a final test claim.

### Two-backbone TSRS evidence from R320

| Backbone | Method | Dice | IoU | B-IoU | SDice@2 | HD95 | ASSD |
|---|---|---:|---:|---:|---:|---:|---:|
| U-Net | Baseline | 0.8666 | 0.7691 | 0.1502 | 0.4688 | 26.63 | 7.37 |
| U-Net | + prior/loss | 0.8735 | 0.7797 | 0.1699 | 0.5166 | 21.86 | 5.81 |
| nnU-Net | Baseline | 0.8971 | 0.8192 | 0.2399 | 0.6545 | 54.34 | 14.17 |
| nnU-Net | + R317 | 0.9029 | 0.8264 | 0.2452 | 0.6684 | 14.01 | 4.05 |

This supports model transfer across two U-Net-family backbones, but not yet broad plug-and-play generality across unrelated advanced architectures.

### RAM: R332 locked test versus R325 native nnU-Net

| Metric | R325 | R332 step2 | Delta | Direction |
|---|---:|---:|---:|---|
| Overall DSC | 0.979362 | 0.979884 | +0.000523 | better |
| Overall IoU | 0.960003 | 0.961000 | +0.000998 | better |
| Overall NSD@2 | 0.888434 | 0.894604 | +0.006170 | better |
| Overall MSD px | 1.307574 | 1.130006 | -0.177568 | better |
| Overlap DSC | 0.870637 | 0.874531 | +0.003895 | better |
| Overlap IoU | 0.772134 | 0.778223 | +0.006089 | better |
| Overlap NSD@2 | 0.780356 | 0.789704 | +0.009349 | better |
| Overlap MSD px | 1.909904 | 1.804907 | -0.104997 | better |
| Pair DSC | 0.828841 | 0.831425 | +0.002584 | better |
| Pair IoU | 0.733197 | 0.738069 | +0.004872 | better |
| Pair NSD@2 | 0.776738 | 0.784847 | +0.008109 | better |
| Pair MSD px | 1.948267 | 1.691503 | -0.256764 | better |
| Overlap RAVD | 0.058705 | 0.059987 | +0.001282 | worse |
| Pair MSD fail rate | 0.003846 | 0.007692 | +0.003846 | worse |

R332 passes the main requirement: area metrics improve slightly while overlap/surface metrics improve more clearly. The two adverse metrics must remain visible.

## 5. Experiment Ledger

| Range / run | Main question | Outcome | Current use |
|---|---|---|---|
| R240-R248 | Can GT-free rules safely select test-time cuts? | No; oracle good, learned/scalar selectors unsafe | Closed as deployment route; preserves oracle evidence |
| R254-R255 | Can generic loss/close-gap PCR fix bridges? | Mixed/no-go | Diagnostic only |
| R256-R258B | Can development and relation heatmaps identify seams? | Signal exists; geometry redundant; age/sex harms specificity | Image/center pathway retained; age/sex removed |
| R259 | Short nnU-Net integration | Immature and invalid for quality comparison | Engineering diagnostic only |
| R260 | Mature nnU-Net + frozen prior | Positive TSRS result with recall trade-off | Historical positive anchor |
| R265-R272 | Anatomical ROI/relation/absolute seam/instance supervision | Learnable, but targeting and over-suppression problems | Mechanism evidence; not final models |
| R275 | Mature paired TSRS control | Canonical control for local fusion studies | Baseline record |
| R279-R283 | Support weighting/gating/GradNorm | Mostly no-go; ASSD improved but Dice/gap or Recall failed | Do not resume global support sweeps |
| R284b | Compact U-Net overlap prior on RAM | Large positive pilot | Shows weak-backbone potential |
| R285 | Same simple idea on nnU-Net | No-go | Motivates local/pair-specific design |
| R286-R288 | Pair/local boundary relation priors | Small positive validation signal | Design precursor |
| R293 | Explicit dual branch on TSRS | No-go; global residual caused most degradation | Avoid global residual correction |
| R307 | Exact R260 binary vs indexed-instance auxiliary | Imperfect instance labels did not help | Confirms label-noise caution |
| R308-R315 | Manual-clean keep-list and prior refit | Better large-error/separation metrics, still conservative | Secondary dataset-quality evidence |
| R316/R316b | Hard-negative safe selector | No-go due recall-specificity conflict | Closed selector form |
| R317 | Continuous image-centers-only TSRS prior | Pass on original-val | Current TSRS anchor |
| R320 | Paper-style paired tables and alpha audit | Positive U-Net and nnU-Net rows; single seed | Current TSRS summary tables |
| R323-R325 | Native RAM seam/overlap and resolution studies | Seam prior weak; mature native nnU-Net established | Baseline and design calibration |
| R326 | Layer-projection output prior | Tiny gains; not better than generic adapter locally | Insufficient mechanism evidence |
| R327-R329 | Error-shaped instance completion | R329 beats baseline and generic adapter | Strong mechanism precursor |
| R330 | Surface/volume regularization | Improves official RAM metrics | Positive validation method |
| R331 | Five-seed R330 stability | Improvements stable across 5 seeds | Robust validation evidence |
| R332 | Joint iterative completion | Positive locked RAM test | Current RAM anchor |
| R333 | Direct combined seam + overlap training | Validation no-go in current implementation | Redesign adapters; do not test |

## 6. Failures and Lessons

1. A correct oracle cut does not imply a deployable cut selector. Foreground-safety must be learned or gated without GT at inference.
2. A seam loss alone tends to contract foreground. Support terms can reverse contraction but may fill the seam; global scalar balancing is unstable.
3. Age/sex contains some TSRS signal but is not universal and increased false-separation pressure. It should not be in the shared core.
4. Compact U-Net improvements do not guarantee gains on nnU-Net. Strong baselines require local, pair/instance-aware, residual corrections.
5. TSRS color identity is noisy; RAM multi-label identity is trustworthy. They require different state adapters even inside one framework.
6. Native resolution and metric tolerance matter. NSD@2px and pixel distances cannot be compared across resized and native pipelines without normalization.
7. Iterative correction can help, but correction depth must be selected on validation and locked before test.
8. EMA pseudo-Dice is insufficient as the sole checkpoint metric when the goal is boundary/interface behavior. Selection needs Dice/IoU hard gates plus boundary/surface criteria.

## 7. Current Method Definition

The cleanest unified formulation is not one unconditional prior. It is:

1. Shared global-local structural representation at the native model resolution.
2. Interaction-state adapter:
   - TSRS: reliable background seam plus uncertain masking;
   - RAM: true multi-instance overlap from independent labels.
3. Conditional actions:
   - input-side seam probability for separation evidence;
   - output-side per-instance completion/refinement for overlap evidence.
4. Conditional loss with non-interference:
   - ordinary segmentation loss;
   - separation loss only on confirmed seams;
   - completion/surface loss only on confirmed overlap or error-shaped instance regions;
   - consistency/distillation term to prevent mature-backbone degradation.

The shared innovation is interaction-aware prior conditioning and non-conflicting optimization. The two datasets do not need identical masks or identical loss weights; they need the same state-conditioned mathematical framework.

## 8. What Changed Our Belief

- Strengthened: spatial priors and interface-specific losses can improve both ordinary segmentation and difficult interaction regions.
- Strengthened: output-side iterative completion is effective for RAM true overlap.
- Weakened: age/sex should be a central universal prior.
- Rejected: one global background-separation rule can handle both background seams and projected overlap.
- Rejected: simply increasing lambda or adding a global residual branch will create larger useful gains.
- Unresolved: whether a protected, frozen-backbone R333 can preserve both R317 and R332 improvements simultaneously.

## 9. Next Actions

1. Preserve R317 and R332 as immutable anchors.
2. R333-TSRS repair: reuse R317's exact nnU-Net preprocessing, logits and native prediction space; do not resize independently to 512.
3. R333-RAM repair: freeze R332/R325 backbone initially; train only a lightweight seam adapter and add R332-output distillation/non-degradation loss.
4. Select on validation with hard Dice/IoU gates, then boundary/surface/interaction metrics. Do not use EMA pseudo-Dice alone.
5. Run ablations separately: shared core only; separation state only; overlap state only; full conditional system.
6. Do not use TSRS clean-test-v2 or RAM test again until all design choices are locked.
7. After validation passes, add paired per-case confidence intervals and multi-seed verification before strong paper claims.

## 10. Artifact and Reproducibility Index

- TSRS R260: `research-workflow/refine-logs/R260_MATURE_NNUNET_PRIOR_SPEC.md`
- TSRS R317 selection: `outputs/analysis/r317_image_centers_only_checkpoint_selection.json`
- TSRS paper tables: `research-workflow/refine-logs/R320_PAPER_READY_EXPERIMENT_TABLES.md`
- RAM official validation audit: `research-workflow/refine-logs/R329_RAM_OFFICIAL_METRICS_VALIDATION_RESULT.md`
- RAM multi-seed result: `research-workflow/refine-logs/R331_RAM_R330_MULTISeed_RESULT.md`
- RAM locked test: `research-workflow/refine-logs/R332_RAM_LOCKED_TEST_VS_R325_RESULT.md`
- R333 plan: `research-workflow/refine-logs/R333_DUAL_INTERACTION_PRIOR_EXPERIMENT_PLAN.md`
- R333 scripts: `scripts/r333_dual_interaction_prior.py`, `scripts/train_r333_ram_dual_interaction_prior.py`, `scripts/train_r333_tsrs_dual_interaction_prior.py`

## Evidence Boundary

- R317 is TSRS original-val, single-seed development evidence.
- R331 is RAM validation multi-seed evidence.
- R332 is a one-time locked RAM test result with point estimates; paired CIs are not yet reported.
- R333 remote result artifacts are not currently synchronized locally, so its exact per-epoch tables should not be treated as a durable final record until copied and audited.
- No claim in this report authorizes clean-test-v2 tuning or mixing TSRS Articular-Surface data.
