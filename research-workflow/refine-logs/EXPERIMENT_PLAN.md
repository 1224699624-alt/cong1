# Experiment Plan

**Problem**: Improve epiphysis segmentation until clean-test-v2 Dice exceeds original ARAA by `+0.02`, target `> 0.9317660066557425`.

**Current Best Valid Result**: R110 R100+R108 patch/CNN arbitrator, clean-test-v2 Dice `0.9177231563529792`.

**Evaluation Rule**: Success claims use only `TSRS_RSNA-Epiphysis_clean_test_v2/test`. The new reannotated test split is excluded from success decisions.

**Date**: 2026-06-28

## Decision State

R110 slightly improved the valid best, but R111 showed that adding R110 to the candidate pool does not create new oracle headroom. R117 tested a pretrained DINOv3 source with explicit instance-separation heads; it improved over weak direct sources in validation structure, but its final clean-test-v2 Dice remained far below R110 and its usable complementarity was tiny.

| System / Oracle | clean-test-v2 Dice |
| --- | ---: |
| R038 anchor | `0.914357` |
| R090 patch arbitrator | `0.916440` |
| R095b R036 patch/basic | `0.916641` |
| R100 R095b+R097 patch/basic | `0.917618` |
| R108 DINOv3 direct source | `0.880116` |
| R110 R100+R108 patch/basic | `0.917723` |
| R117 DINOv3 instance-separation source | `0.894567` |
| R118_fast per-image oracle with R117 added | `0.919153` |
| R118_fast vote_ge_5 with R117 added | `0.917669` |
| R118_fast GT-leaking pixel oracle | `0.956813` |
| best non-leaking R038/R090/R095b/R097/R100/R106/R108/R110 combo | `0.917039` |
| per-image oracle | `0.919143` |
| GT-leaking pixel oracle | `0.963184` |
| target | `0.931766` |

Interpretation: R110 remains the valid best, with a target gap of `0.014043`. R117 is not a useful standalone model (`0.894567`) and R118_fast selects it in only `2/81` per-image oracle cases. The apparent headroom is again GT-leaking pixel-level complementarity, not a deployable readout. Do not run an R117 patch/readout rescue.

## Updated Thesis

The same-family postprocessing routes are mostly closed:

- R092 closed R038/R089/R090/R091 candidate disagreement.
- R094 closed ROI instance redraw complementarity.
- R096 closed the R036+R090 patch complement route.
- R101 shows R097 adds real pixel-level diversity, but non-leaking combinations are still far below target.

R097 and R108 are not standalone replacements, but they are useful diverse image-model sources. R100 and R110 are valid gains from exploiting them, yet the gain scale is now around `1e-4` to `1e-3`, far below the remaining target gap. The next valid attempt must change the mechanism, not merely the anchor/candidate threshold, local_stats setting, or same patch CNN.

## Active Run

| Run ID | Purpose | Train/Tune | Final Evaluation | Baseline | Status |
| --- | --- | --- | --- | --- | --- |
| R108 | DINOv3 ConvNeXt-tiny direct source | original train/val direct segmenter | clean-test-v2/test and original-test control | R100 Dice `0.917618` | DONE_NEGATIVE_COMPLEMENTARY |
| R110 | R100+R108 patch readout | original train/val, R100-like anchor + R108 candidate | clean-test-v2/test, R100 anchor + R108 candidate | R100 Dice `0.917618` | DONE_NEW_BEST_TINY |
| R111 | R110 complementarity audit | non-GPU oracle | clean-test-v2/test | target Dice `0.931766` | DONE_PATH_BLOCK |
| R112 | DINOv3 distance-field source | original train/val direct segmenter with signed-distance supervision | clean-test-v2/test and original-test control | R108 Dice `0.880116` | DONE_NEGATIVE_COMPLEMENTARY |
| R113 | R112 complementarity audit | non-GPU oracle | clean-test-v2/test | R111 pixel oracle `0.963184` | DONE_MIXED |
| R114 | R100+R112 patch readout | materialized R112 train/val, then patch/CNN readout | clean-test-v2/test, R100 anchor + R112 candidate | R110 Dice `0.917723` | DONE_FLAT_NEGATIVE |
| R115 | precision-controlled SAM adaptation | original train/val SAM ViT-B decoder/GBC/prompt-encoder tuning with structured foreground/background prompts | clean-test-v2/test and original-test control | R110 Dice `0.917723` | DONE_NEGATIVE |
| R116 | R115 complementarity audit | non-GPU oracle over R110/R100/R115 | clean-test-v2/test | R110 Dice `0.917723` | DONE_MIXED |
| R117 | pretrained structural source model | DINOv3 ConvNeXt-tiny FPN plus instance-separation auxiliary heads | clean-test-v2/test and original-test control | R112 Dice `0.888406`, R110 Dice `0.917723` | DONE_NEGATIVE_COMPLEMENTARY |
| R118_fast | R117 complementarity audit | non-GPU fast oracle over R038/R090/R095b/R097/R100/R110/R117 | clean-test-v2/test | target Dice `0.931766` | DONE_MIXED |
| R119 | high-resolution contour/source test | DINOv3 high-resolution direct source, no patch/readout rescue | clean-test-v2/test and original-test control | R110 Dice `0.917723` | DONE_NEGATIVE |

## Decision Gates

R110 exceeded R100 only by `0.000105`, and R111 showed no new pixel-oracle headroom from R110 itself. R112 then showed that distance-field supervision improves a direct DINOv3 source (`0.888406` vs R108 `0.880116`) and slightly raises the pixel oracle (`0.964421`), but R114 failed to recover it (`0.917715`, just below R110). The next experiment must not be another threshold/local-stats/identical patch sweep.

1. Close R097/R108 direct-source plus R090-style patch readout variants unless a genuinely new readout mechanism is introduced.
2. Close the current distance-field direct-source plus same patch/CNN readout path. It is useful scientifically but not enough for the target.
3. Prefer one of two remaining different mechanisms: SAM-adapter ROI retraining with a precision control that avoids the R055/R057 collapse, or a label/layout-informed training objective that changes the base predictor and explicitly handles anatomical instance separation.
4. Keep R110 as the current valid best until a later run exceeds Dice `0.9177231563529792` on `TSRS_RSNA-Epiphysis_clean_test_v2/test`.

R115 chooses the first route: a SAM ViT-B adaptation with explicit foreground/background point prompting, prompt-location supervision, boundary contrastive regularization, and a conservative mask-to-prompt refinement pass. This is not another R090-style local patch/readout sweep; it changes the prompted SAM mask source itself and then evaluates directly on clean-test-v2.

R115 failed as a direct source and R116 shows it is not useful for whole-image selection. R117 also failed as a deployable structural source: clean-test-v2 Dice `0.894567`, original-test Dice `0.878650`, and R118_fast shows only tiny per-image utility. R119 then tested high-resolution DINOv3 distance supervision at `img-size 768`, but clean-test-v2 Dice was only `0.887952`, slightly below R112 and far below R110. Close direct-source resolution scaling.

Next mechanism should not be another direct DINOv3/Swin/SAM source or R090-style patch readout. The remaining plausible route is a non-GPU/low-GPU target-error audit that identifies whether R110's remaining gap is dominated by systematic label/layout cases, then only launch a new experiment if the audit yields a concrete, non-leaking rule or training-data variant.

R120 completed that audit around R110. R110's remaining clean-test-v2 errors are bridge/layout-heavy: `56/81` bridge-like cases, `29/81` with Boundary IoU `<0.20`, and `15/81` with Dice `<0.90`. Adding the weak diverse sources creates a disagreement region covering only `0.009729` of pixels but containing `0.513520` of R110 errors; a GT-leaking disagreement oracle reaches Dice `0.960361`. This is a strong diagnostic signal, but it is clean-test-v2-only and cannot be used directly for a deployable rule. The next step must test whether the same signal exists on original validation with a trainable/non-leaking readout, not tune thresholds on clean-test-v2.

R121 tested that transfer condition on original validation using the existing R100-like train/val anchor plus R097/R112 candidates. The result was negative: val anchor Dice was only `0.895547`, candidate disagreement covered `0.004319` pixels, captured only `0.198109` of anchor errors, and the GT-leaking disagreement oracle reached only `0.917539`. Therefore the clean-test-v2 disagreement headroom is not represented strongly enough in the original-val training signal. Do not launch another disagreement MLP/CNN/stack readout trained on this proxy.

Current ARIS decision: the bottleneck is no longer architecture search alone. The useful clean-test-v2 errors are localized, but the available original train/val proxy does not teach a stable non-leaking readout. Further GPU experiments should pause until a validation/data strategy is revised, for example by constructing an isolated hard-case validation audit set, verifying clean-test-v2 hard cases visually, or obtaining/curating train-side examples that match the R110 bridge/boundary failure mode.

R122 created that visual audit pack at `outputs/analysis/r122_r110_hard_case_audit_pack/`. The 24 selected hard cases have mean Dice `0.893097`, mean Boundary IoU `0.203074`, and mean component_delta `-2.458`, confirming that the dominant visible pattern is under-separated components/bridges plus boundary misses. Visual spot checks show at least three mixed modes: low-contrast wrist/carpal bridge cases where all candidates fail similarly, small epiphysis boundary misses where candidate disagreement can help, and recall-limited cases where R110 misses narrow label extent. This mixture explains why a single proxy-trained readout has not transferred.

Next actionable step: manually tag the 24 R122 panels into failure modes (`low_contrast_bridge`, `boundary_shift`, `missing_small_component`, `possible_label_ambiguity`, `candidate_fixable`) and only then decide whether to create a small isolated data/label protocol variant or request/derive additional train-side hard examples.

R123 initialized that review layer in `outputs/analysis/r122_r110_hard_case_audit_pack/R123_HARD_CASE_TAXONOMY.md` plus JSON/CSV templates. Auto-tag counts are mixed: `candidate_fixable=7`, `candidate_shared_failure=6`, `under_separated_bridge=6`, `overmask_low_precision=7`, `severe_boundary_shift=6`, and `underreach_high_precision=4`. Decision gate: do not launch R124 training unless manual review confirms a dominant, trainable mode. If `candidate_fixable` is not clearly dominant after review, the next step should be data/label protocol work, not architecture/readout work.

R124 completed the next non-GPU data step: `outputs/analysis/r124_hard_case_trainval_retrieval/` ranks original train/val images by similarity to the R122/R123 hard-case prototypes using label layout, component spacing, boundary-shape proxies, and local image contrast features. The top 80 candidates contain `74` train and `6` val cases, so train-side analogues do exist. However, their nearest prototypes are mostly `underreach_high_precision` (`25/80`) and `under_separated_bridge` (`17/80`), with only `2/80` nearest to `candidate_fixable`. This supports hard-case curation / label-protocol review before another GPU fusion/readout experiment. The immediate next gate is visual review of the R124 top panels and, if confirmed, creation of an isolated hard-case curation manifest rather than any change to the original dataset.

R125 converted R124 into a concrete curation artifact at `outputs/analysis/r125_hard_case_curation_manifest/`. Visual spot checks of the R124 top panels confirmed train-side analogues of low-contrast wrist/carpal multi-component cases and bridge/boundary-fragment modes. The manifest selects `64` hard-train cases and `6` hard-val-audit cases, with hard-train prototypes led by `underreach_high_precision=18`, `under_separated_bridge=16`, and `candidate_shared_failure=9`; `candidate_fixable` is only `1/64`. This is a training-data strategy artifact, not a model result. If continuing to GPU, the next justified step is an isolated R126 hard-case sampling or label/layout-protocol experiment that uses this manifest for train-time sampling and keeps clean-test-v2/test solely as final evaluation.

R126 is that narrow GPU test. It kept the R117 DINOv3 instance-separation architecture fixed and changed only train-time sampling: `outputs/analysis/r125_hard_case_curation_manifest/r125_hard_case_curation_manifest.json` supplied `64` hard-train names, all matched on remote, with sampler weight `3.0` and epoch multiplier `1.25`. The run completed after 36 epochs. Best validation Dice was epoch 33 at `0.867308`, close to but still below R117's best validation Dice (`0.869870`). Final clean-test-v2 Dice was `0.887837` with Precision `0.847248`, Recall `0.934430`, Boundary IoU `0.166387`, component-count error `2.728395`, and false-bridge flag `0.839506`; original-test Dice was `0.872894`. This is negative relative to R117 (`0.894567`), R110 (`0.917723`), and the target (`0.931766`).

R126 interpretation: oversampling the R125 hard-train pool increases recall pressure but worsens precision, boundary quality, and bridge behavior. Because R125 contains mostly shared-failure / bridge / underreach analogues rather than candidate-fixable cases, sampler emphasis alone appears to amplify difficult anatomy without teaching a clean separating rule. Do not launch R127 as another weight-only sampler sweep (`1.5`, `2.0`, `3.0`) unless a new audit shows a specific subset or label correction that changes the supervision signal. The next ARIS step should be non-GPU or low-GPU: inspect the R125 selected labels/panels for label-protocol ambiguity, build a small isolated corrected/verified hard-case manifest, or seek a new literature mechanism that explicitly models ordered epiphysis instances with stronger separation priors rather than simple resampling.

R127 performed that non-GPU protocol audit at `outputs/analysis/r127_hard_case_protocol_audit/`. It compared the R125 hard pool against full train/val robust layout/image statistics and checked how R126 behaved on the 24 R122 hard clean-test-v2 cases. The hard pool is not a broad label-outlier pool: severe robust outliers at top3-z `>=3.5` were `0/70` across hard-train plus hard-val-audit, and the hard-train prototype mix has only `1/64` `candidate_fixable` case. R126 degraded the R122 hard cases to mean Dice `0.868973`, Boundary IoU `0.151140`, component-count error `3.416667`, and false-bridge `0.833333`. Decision gate: no broad label filtering, no further simple resampling, and no next GPU run without new labels/protocol evidence. The next productive action is either manual review/correction of the R122/R125 panels or a genuinely new ordered-instance anatomical model that supplies supervision beyond the current binary label weighting.

R128 tested whether the instance-valued labels can directly support that ordered-instance model. The answer is mixed but mostly a gate against naive slots: train/val labels have contiguous IDs only `0.840371` of the time, component count varies from `2` to `29` with q10/median/q90 `23/27/28`, id-vs-PC1 order inversion median/q90 is `0.156695/0.192029`, and a single principal axis explains only median `0.635144` of component layout variance. Therefore R078-style fixed slots and any one-dimensional left-right/top-bottom ordering should stay closed. A possible new architecture, if pursued, must be variable-slot and set/matching based, e.g. a DETR/Mask2Former-like query decoder or permutation-invariant instance mask head with Hungarian matching plus binary union loss. That would be a genuinely new mechanism, but it should be scoped as a high-risk R129 path, not a quick continuation of prior slot models.

R129 implemented the smallest viable version of that high-risk path: a lightweight U-Net predicts `32` query masks plus a binary union mask; query masks are matched to instance-valued labels with Hungarian assignment during training, so label ID and spatial order are not assumed. Local CPU smoke and remote GPU smoke verified the full train -> val selection -> clean-test-v2 inference path, but the full run failed decisively. Best validation was epoch 13 Dice `0.821461`, Precision `0.724573`, Recall `0.963986`, Boundary IoU `0.294907`. Final clean-test-v2 Dice was `0.840953`, Precision `0.744594`, Recall `0.970739`, Boundary IoU `0.096504`, false-bridge `0.851852`; original-test Dice was `0.824322`. This is `-0.076771` below R110 and `-0.090813` below the target. The failure mode is not capacity-limited in an encouraging way: it is severe high-recall / low-precision overmasking with bridge-heavy unions. Close the lightweight set-matching source-model branch; do not run blind query-count, threshold, or blend sweeps.

R130 should be a non-GPU decision artifact before any new training launch. The evidence now closes simple resampling, broad filtering, fixed slots, single-axis ordering, lightweight set matching, and same-family patch/readout sweeps. The next valid direction must either (1) produce stronger label/protocol evidence through manual review or isolated corrected hard-case manifests, or (2) start from a substantially different literature-backed architecture whose training signal explicitly prevents bridge-like unions, such as boundary-first instance separation, graph/layout-constrained posteriors, or a stronger Mask2Former/DETR-style query decoder with explicit no-object/background calibration and separation losses. Until that decision artifact exists, keep R110 as the valid best and do not spend GPU on R130.

R130 chooses the architecture route but keeps it scoped to the existing verified training chain before any heavy Mask2Former/Mask DINO rewrite. The concrete minimal mechanism is `r130_dinov3_bridge_suppressed_instance_sep`: reuse the R117 DINOv3 ConvNeXt-tiny instance-separation source, then add disabled-by-default bridge-gap/background precision losses in `scripts/train_timm_instance_separation_segmenter.py`. R130 turns those switches on with stronger separation and boundary weighting, a wider separation band, and a conservative threshold/suppression validation grid. This directly tests the R129/R126 failure mode: whether explicit penalties on predicted foreground in inter-instance separation bands and background can reduce bridge-heavy overmasking without collapsing recall.

R130 is not claimed as full Mask2Former/Mask DINO. It is a low-risk bridge-prevention probe that must pass a smoke run before any full GPU launch. If it cannot improve original-val precision/false-bridge behavior relative to R117/R126, the next architecture attempt should not be another DINOv3 instance-separation retune; it should either move to a real query/no-object decoder or return to label/protocol correction.

R130 completed with a mixed-negative result. Its validation trajectory was encouraging for structure: best val epoch 34 reached Dice `0.876224`, Precision `0.849432`, Recall `0.912570`, Boundary IoU `0.462412`, and false-bridge `0.208333`, exceeding R117 validation Dice. But the final clean-test-v2 result was only Dice `0.890938`, Precision `0.855765`, Recall `0.931193`, Boundary IoU `0.180672`, false-bridge `0.444444`; original-test Dice was `0.876247`. This confirms that explicit bridge/background penalties can make the source structurally cleaner, but not competitive with R110.

R131 then audited whether R130 is at least a useful complementary candidate. It is not useful enough for a GPU fusion run: R130 was selected on `0/81` per-image oracle cases, per-image oracle over the expanded pool reached only Dice `0.919179`, and the best non-leaking vote (`vote_ge_6`) was `0.916422`. The pixel oracle Dice `0.969043` remains GT-leaking and therefore does not justify another patch/readout learner by itself. Current decision: close R130 as a source/readout branch. The next experiment should not be another DINOv3 instance-separation loss-weight sweep; it must either implement a real query/no-object decoder with stronger calibration or move back to label/protocol correction.

R132 chooses the query/no-object route while keeping the implementation scoped to the existing script/launcher style. `scripts/train_timm_set_instance_segmenter.py` combines the verified R117/R130 timm DINOv3 ConvNeXt-tiny FPN feature extractor with R129-style Hungarian query-mask supervision, then adds explicit query objectness / no-object calibration and a binary union auxiliary head. This is meant to test whether R129's overmasking came from weak features and absent no-object suppression rather than from the set-matching formulation itself. The smoke gate must complete train -> val -> clean-test-v2 -> original-test before any full R132 launch. If full R132 does not beat R110 on clean-test-v2, do not continue with blind query-count or threshold sweeps.

R132 completed negative. Best validation was epoch 29 Dice `0.866350`, Precision `0.808439`, Recall `0.944858`, Boundary IoU `0.431940`, false-bridge `0.697917`, which did not reach R130's validation Dice `0.876224`. Final clean-test-v2 Dice was only `0.873978`, Precision `0.814021`, Recall `0.945500`, Boundary IoU `0.142318`, component-count error `3.209877`, false-bridge `0.851852`; original-test Dice was `0.858992`. Therefore the lightweight query/no-object formulation did not solve the R129 failure. It improved R129 but remained far below R110 and target. Do not run blind query-count, threshold, blend, or loss-weight sweeps on this script. The next step should return to non-GPU evidence: identify whether valid progress now requires label/protocol correction, or a much stronger externally validated architecture with a different supervision/readout path.

R133 made that branch decision in `research-workflow/refine-logs/R133_DIRECTION_DECISION.md`. The literature-backed distinction is important: Mask2Former and Mask DINO are not simply query heads on an FPN mask map; they use masked/deformable attention, detection-style queries/denoising, and high-resolution pixel embeddings. R132 closes only the lightweight approximation, not the full family. Given the repeated failures of R065-R132 lightweight architecture routes and the localized mixed hard-case evidence from R122-R128, the recommended next step is `R134-data`: build an isolated label/protocol hard-case correction manifest and train/val variant. Only choose `R134-arch` if the project is ready to pay the engineering cost of a full external Mask2Former/Mask DINO integration.

R134-data completed the non-GPU manifest step at `outputs/analysis/r134_label_protocol_review_manifest/`. It separates clean-test-v2 failures from train/val review candidates: `24` clean-test-v2 cases are diagnostic references only, `64` train cases are eligible for isolated corrected/verified variants after human review, and `6` validation cases are reserved for protocol audit rather than final success. The train review queue has `25` P0 bridge/boundary cases, `38` P1 cases, and `1` P2 case. This preserves the evaluation rule: clean-test-v2 remains untouched and is not used for training, tuning, or manifest gating. A later GPU run is gated on human decisions, with a proposed minimum of `20` reviewed train cases and at least `8` confirmed corrections or protocol-clear hard examples.

R134-gate made that review gate executable. `scripts/check_label_protocol_review_manifest.py` validates allowed human decisions, verifies clean-test-v2 rows remain diagnostic-only, checks panel paths, cross-checks CSV row count, and reports whether the manifest can move to an isolated corrected/verified variant. The current status file `outputs/analysis/r134_label_protocol_review_manifest/r134_review_gate_status.json` has `gate_pass=false`, `errors=[]`, `warnings=[]`, `reviewed_train=0`, and `confirmed_train_for_variant=0`. This means the data branch is ready for review, not ready for GPU.

R135-build-prep added the safe follow-up builder `scripts/build_label_protocol_dataset_variant.py`. It copies the original Epiphysis dataset into `data/raw_variants/TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1` only after the R134 gate passes, and it optionally replaces reviewed train/val labels from a correction directory. It refuses to use clean-test-v2 rows for variant creation and never edits `data/raw`. Current dry-run correctly fails the gate (`reviewed_train=0`, `confirmed_train=0`) and does not create the target dataset. This prepares the next step without bypassing human review.

R135-decision-apply completes the review-ingestion side of the same gate. `scripts/apply_label_protocol_review_decisions.py` reads a reviewer-filled CSV and updates only `human_decision`, `human_notes`, and `review_status` in the R134 JSON manifest. It defaults to ignoring clean-test-v2 decisions so diagnostic reference cases cannot leak into train/variant decisions, while still allowing train/val review rows to drive the later gate. Local verification covered three cases: the untouched real manifest still reports `reviewed_train=0`; a small temporary CSV produced `reviewed_train=2`, `reviewed_val=1`, and ignored one clean-test-v2 decision; a synthetic 20-train CSV produced checker `gate_pass=true` and builder `dry_run_ok` without mutating the true manifest. The CSV reader now uses `utf-8-sig`, so PowerShell/Excel-style BOM CSV files do not break the `image` column. The branch is now waiting on real human decisions, not code plumbing.

R136-review-package turns the R134 manifest into a concrete human-review work package at `outputs/analysis/r134_label_protocol_review_manifest/review_package/`. `scripts/prepare_label_protocol_review_package.py` writes a UTF-8-BOM train/val worklist CSV, a separate clean-test-v2 reference-only CSV, a Markdown review checklist, a static HTML review page, and a machine-readable summary. Verification confirms the worklist has exactly `64` train rows and `6` val audit rows, the reference file has `24` clean-test-v2 rows, the HTML page has `94` case cards (`70` train/val plus `24` clean-test references), filter controls, and resolvable panel image links. This step intentionally does not fill any decisions; the real gate still reports `reviewed_train=0` and `gate_pass=false`. Its purpose is to make the next human action precise: review P0/P1 train rows first, mark at least 20 train decisions, and keep clean-test-v2 diagnostic examples separate.

R137-review-ui-export makes the same review package directly actionable in the browser. The static HTML page now gives each train/val card a `human_decision` dropdown and notes input, keeps all clean-test-v2 cards locked as reference-only, and exports `r134_train_val_review_worklist_reviewed.csv` with the columns expected by `scripts/apply_label_protocol_review_decisions.py`. Local verification found `94` total cards, `70` decision selectors, `70` notes inputs, `24` locked reference cards, and the export button wired to the reviewed CSV filename. The real manifest and gate remain unchanged (`reviewed_train=0`, `gate_pass=false`), so this adds review ergonomics without fabricating labels or altering the dataset.

R138-review-preview adds a read-only guard between browser export and manifest mutation. `scripts/preview_label_protocol_review_decisions.py` loads the reviewed CSV with UTF-8-BOM support, checks each row against the manifest index and allowed decision schema, skips clean-test-v2 decisions by default, and reports `gate_pass_if_applied` plus train/val decision counts. Local smoke tests covered five cases: empty worklist (`gate_pass_if_applied=false`), 10 train decisions (`false`), 20 confirmed train decisions (`true`), invalid decision plus unknown image (`status=error`), and clean-test-v2 mix-in (`warning`, skipped). The actual R134 manifest is still unchanged and the real checker still reports `gate_pass=false`, `reviewed_train=0`.

R139-review-pipeline-wrapper adds the safe execution wrapper for after review. `scripts/run_label_protocol_review_pipeline.py` chains preview, application to a temporary manifest, checker, and dataset-variant builder dry-run by default. It requires explicit `--apply-in-place` before mutating the real manifest and explicit `--create-variant` before creating `data/raw_variants/TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1`. Local verification showed the empty worklist is rejected at preview with `mutated_manifest=false`, while a temporary 20-train gate-pass CSV reaches `dry_run_ok` through a temporary manifest: checker passes, builder reports source split copy counts `875/96/97`, and clean-test-v2 remains diagnostic-only. The true manifest and data directories remain unchanged.

R140-launch-prep prepares the first post-gate GPU run without launching it. `outputs/bridge_logs/run_r140_reviewed_variant_instance_sep.sh` is a guarded DINOv3 instance-separation retrain on `data/raw_variants/TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1`, using the structurally cleaner R130-style loss settings. It refuses to run unless the reviewed variant directory and `label_protocol_variant_metadata.json` exist, re-runs the R134 checker, and verifies that the variant metadata states clean-test-v2 was diagnostic-only. Its final success evaluation remains fixed to `TSRS_RSNA-Epiphysis_clean_test_v2/test`, while original `TSRS_RSNA-Epiphysis/test` is only the control metric. `scripts/check_r140_reviewed_variant_launcher.py` statically verifies the launcher tokens and forbids `flat_output_suffix`, `reannotated`, or clean-test as the training dataset. This makes the next GPU step ready but still gated on real human review and isolated-variant creation.

R141-r140-monitor adds the post-run judgment step for R140. `scripts/monitor_r140_reviewed_variant_result.py` reads `outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_clean_test_v2_metrics.json` when it exists, compares `mean.dice` to the target `0.9317660066557425` and current valid best R110 `0.9177231563529792`, and writes `outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_result_summary.json`. If metrics are absent it reports `waiting_for_metrics` and, if a history file exists, the epoch count and best validation Dice so far. Local tests covered missing metrics, a below-best R130 metrics simulation, and a synthetic target-met metrics file. The R140 launcher now calls this monitor after training, so a completed R140 run will immediately produce an actionable status: `target_met`, `new_best_below_target`, or `below_best`.

R142-status-snapshot was refreshed after the human review and R140 run. The R134 gate now passes (`64` train reviewed, `54` confirmed), the isolated variant `TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1` exists, and R140 has a completed clean-test-v2 monitor summary with status `below_best`.

R140 tested the reviewed-data branch by retraining the R130-style DINOv3 instance-separation source on the isolated reviewed variant. It did not improve the valid best. Best validation was epoch 35 Dice `0.876365`, but final clean-test-v2 Dice was only `0.885500`, Precision `0.854157`, Recall `0.920853`, and Boundary IoU `0.174439`. This is `-0.032223` below R110 (`0.917723`) and `-0.046266` below the target (`0.931766`). The original-test control JSON has no evaluated cases and should not be used for claims; clean-test-v2 metrics are the valid decision signal.

R141 audited R140 complementarity against R110. R140 is better than R110 on only `2/81` clean-test-v2 cases, with tiny gains on `3503.png` (`+0.002221`) and `8435.png` (`+0.001550`). The R110/R140 per-image oracle Dice is `0.917770`, only `+0.000047` over R110. Therefore R140 is not a useful fusion/readout branch. Close reviewed-variant source training unless genuinely corrected label PNGs or a different model family are introduced.

Post-R140 decision: the reviewed-data branch, as currently instantiated, is exhausted. The next useful ARIS step is not R140 thresholding, R140 fusion, or another DINOv3 instance-separation loss sweep. R143 should be a direction decision and smoke-gated implementation plan for a genuinely new literature-backed architecture.

R143 candidate directions:

1. Full external Mask2Former / Mask DINO style integration: real masked/deformable attention, high-resolution pixel embeddings, detection-style queries/no-object handling, and instance supervision. This is the most faithful literature pivot, but remote dependency readiness is currently poor: `torch`, `torchvision`, `timm`, and `segment_anything` exist, while `detectron2`, `mmdet`, `mmcv`, `mmengine`, `transformers`, `monai`, and `nnunetv2` are absent.
2. Medical segmentation recipe pivot: install or implement a robust nnU-Net-like high-resolution 2D baseline with strong augmentation, deep supervision, test-time augmentation, and connected-component/layout postprocessing. This is less novel, but it tests whether the gap is caused by underpowered training recipe rather than architecture family.
3. Anatomical graph/layout postprocessing: only justified if a new non-leaking train/val diagnostic shows layout constraints can improve Dice without the recall loss seen in R074/R075. Current evidence does not support launching this first.

Recommended next action: create `R143_DIRECTION_DECISION.md`, choose one branch, and define a smoke test before any long GPU run. If choosing external Mask2Former/Mask DINO, first run an environment/install probe and a 2-image train/infer smoke on one free GPU. If choosing the medical recipe pivot, first run a tiny high-resolution 2D U-Net/nnU-Net-style smoke in the existing PyTorch environment. Keep success evaluation fixed to `TSRS_RSNA-Epiphysis_clean_test_v2/test`.

R143 implemented and ran the immediate medical-recipe pivot. `scripts/train_timm_unet_segmenter.py` gained R143-specific switches for stronger X-ray augmentation, EMA, gradient accumulation, TTA-compatible evaluation, and validation-selected small-component cleanup. The smoke run completed train -> val -> clean-test-v2 -> original-test without crash. The full run used DINOv3 ConvNeXt-tiny U-Net/FPN at `768`, strong X-ray augmentation, Dice/BCE plus boundary/distance losses, EMA `0.997`, and validation selection over threshold plus minimum component area.

R143 result: best validation was epoch 39 Dice `0.879224`, Precision `0.816796`, Recall `0.958136`, Boundary IoU `0.314106`, threshold `0.60`, min component area `16`. Final clean-test-v2 Dice was `0.891375`, Precision `0.821758`, Recall `0.975703`, Boundary IoU `0.161392`, false-bridge flag `0.827160`; original-test Dice was `0.877020`. This is far below R110 Dice `0.917723` and the target `0.931766`. The failure mode is high-recall / low-precision overmasking with persistent bridge behavior, so medical-recipe scaling alone does not solve the target gap.

Post-R143 decision: do not run blind R143 sweeps over resolution, EMA, threshold, or augmentation. The validation curve improved only to the same broad source-model band as R130/R140, while clean-test-v2 remained around `0.89`. The next ARIS branch should be R144 full external architecture feasibility: isolate dependencies for a real Mask2Former/Mask DINO style framework, convert the epiphysis instance labels into the required training format, and run a 2-image train/infer smoke before any long GPU run. If dependency integration fails or is too expensive, the project should pause for either true corrected labels or a more substantial architecture import plan rather than continuing local lightweight approximations.

R144 completed the first external-architecture feasibility smoke using HuggingFace `Mask2FormerForUniversalSegmentation`. Dependencies were installed into isolated project path `.tmp/r144_hf_deps`, not into the active conda environment. The smoke converted instance-valued train labels into variable-instance tensors, ran two optimizer steps on two original train images, ran inference on two clean-test-v2 images, exported union masks, and wrote `outputs/analysis/r144_mask2former_hf_smoke_clean_test_v2_metrics.json`. Smoke Dice was near zero (`0.000047`) because the model was tiny/random and trained only two steps; this is not a quality result. The important result is feasibility: the real external Mask2Former code path can consume this dataset without data-format blockers.

Next decision: proceed to R145 small-overfit gate before any full run. R145 should train the HF Mask2Former path on 8-16 train images long enough to verify loss decrease and train-set mask recovery, then do a tiny val/clean-test-v2 diagnostic export. Only if R145 can overfit and produce non-empty useful masks should a full external-architecture GPU experiment be launched.

R145 completed that small-overfit gate but did not pass it. Loss decreased from `28.99` to `15.74` over 120 steps on 16 train images, proving that optimization and supervision wiring are active. However, train, val, and clean-test-v2 exported masks were all empty at threshold `0.35`, giving Dice `0` on every split. R146 then bypassed the class-score foreground gate with mask-only union readout and still produced empty masks on train/val/clean. This means the full external architecture branch is not ready for full training. The next step is R147 logits/threshold diagnostics on the R146 checkpoint: inspect mask probability maxima/quantiles and sweep thresholds to decide whether the model is learning low-confidence spatial masks or failing to learn usable mask logits at all.

R147 answered the readout question. The R146 checkpoint assigns high foreground class probability to the queries (`~0.922`, all 48 queries retained by a 0.05 class gate), but the mask logits are extremely negative: train mean/max union-probability maxima were only `1.30e-05/5.20e-05`, val `9.65e-06/1.31e-05`, and clean-test-v2 `1.03e-05/1.11e-05`. All thresholds from `0.001` to `0.35` still yielded empty masks and Dice `0`. Therefore the empty-mask failure is not caused by class gating or an ordinary threshold setting; the mask decoder output scale is broken for usable foreground masks.

R148 then tested whether the tiny probabilities at least preserve useful spatial ranking. They do not. Ultra-low threshold probing gave best Dice only train `0.148271`, val `0.085552`, and clean-test-v2 `0.188526`. A GT-area top-k oracle, used only as a diagnostic and not as a deployable readout, reached only train `0.355161`, val `0.239494`, and clean-test-v2 `0.268285`, with Boundary IoU around `0.01`. This closes threshold/readout rescue for the R145/R146 checkpoint.

Current Mask2Former decision: do not run a full HF Mask2Former training job, and do not spend GPU on threshold sweeps from the current checkpoint. The next valid step is R149 target/loss wiring rescue: force a 1-2 image train-only overfit, inspect loss components and mask probabilities, and verify high train Dice before any val/clean-test-v2 diagnostic. If R149 cannot overfit one or two images, abandon the current HF integration and either import a more faithful external implementation or pivot away from Mask2Former.

R149/R149b completed the target/loss wiring rescue. The union-target variant proved that the HF model and dependency path are not fundamentally broken: on one train image, mask probabilities recovered from the R146 `1e-5` collapse to `union_max=1.0`, and best train Dice reached `0.811855` with Precision `0.765057`, Recall `0.864752`, and Boundary IoU `0.121593`. This is a meaningful rescue signal, but it failed the predeclared overfit gate of `>=0.85`, and val/clean diagnostic transfer remained poor (`0.017860` / `0.303562`). The instance-target control was weaker: train Dice only `0.650195`, Boundary IoU `0.035831`, val Dice `0.009547`, and clean-test-v2 diagnostic Dice `0.220896`.

Updated decision: do not run full HF Mask2Former instance training. The current instance target path is not reliable enough, and the union path is only partially rescued. If continuing this external architecture family, the next experiment must be R150, a stricter union-source overfit/resolution/loss rescue whose success criterion is train-only overfit quality, ideally Dice `>=0.90` and materially better Boundary IoU on 1-2 train images. Only after that gate passes should any clean-test-v2-scale run be considered. If R150 cannot pass, the HF integration should be closed in favor of a more faithful external implementation or a different literature-backed architecture.

R150 improved the union-source overfit gate. With higher resolution (`384`), slightly larger hidden dimension (`128`), 64 queries, four decoder layers, and stronger mask/dice loss weights, 1-image train Dice reached `0.891038`, with Precision `0.883896`, Recall `0.898297`, Boundary IoU `0.215579`, and stable probability scale. This passes the earlier `0.85` gate and is much better than R149/R149b, but it is still just below the preferred `0.90` gate and the val/clean-test-v2 diagnostics remained weak (`0.166134` / `0.205829`).

Updated R150 decision: the HF union-source path is learnable enough for one more small gate, but not for a full clean-test-v2-scale run. R151 should train the same union-supervised path on a small subset (`8-16` train images) and select only on val. Clean-test-v2 may be evaluated only after the protocol is fixed, as a diagnostic final inference, not as tuning feedback. If R151 validation remains far below the existing source-model band, close the current HF Mask2Former integration.

R151 completed that small multi-image gate and closed the current tiny-random HF route. Training on 16 images reached train Dice `0.857548`, but validation Dice was only `0.710850` with Boundary IoU `0.079828`; the four-image clean-test-v2 diagnostic Dice was `0.639333`. This is far below the existing source-model band and far below R110/target. The result is useful diagnostically: the HF code path can learn something after union supervision, but the random tiny configuration is not a viable source model.

Updated decision after R151: do not run full HF Mask2Former union or instance training, and do not continue tuning the current tiny random HF configuration. The next ARIS step must be R152 direction decision: either import a more faithful external implementation with pretrained weights/backbone and correct Mask2Former/Mask DINO training recipe, or pivot to a different literature-backed architecture. Any next GPU experiment must define a fresh smoke/overfit gate before clean-test-v2 evaluation.

R152 completed that decision artifact at `research-workflow/refine-logs/R152_POST_HF_DIRECTION_DECISION.md`. It closes the current HF tiny-random route and selects the faithful external implementation branch as the next option to probe, not yet to train. The key distinction is that official Mask2Former and Mask DINO implementations are Detectron2-based and use pretrained/high-resolution components and custom ops that the HF tiny random route did not exercise.

Remote dependency status currently shows `torch`, `torchvision`, and `timm` are available, while `detectron2`, `fvcore`, `iopath`, `pycocotools`, `mmcv`, `mmengine`, `mmdet`, `monai`, and `nnunetv2` are unavailable. Therefore R153 must be a non-mutating feasibility probe for isolated official Detectron2-based Mask2Former/Mask DINO setup. Do not install into `yolo-sam-gpu`, and do not launch training until R153 produces a go/no-go artifact.

R153 completed the non-mutating feasibility probe at `research-workflow/refine-logs/R153_FAITHFUL_EXTERNAL_FEASIBILITY_PROBE.md`. The server has two RTX 4090 GPUs, PyTorch `2.5.1+cu118`, CUDA runtime `11.8`, GitHub access, gcc/g++ `13.3`, and about `167G` free disk. However, `nvcc` is not available, `conda` is not on the default shell path, and `detectron2`, `fvcore`, `iopath`, `pycocotools`, and `ninja` are absent. This makes official Detectron2-based Mask2Former/Mask DINO installation high-risk because custom CUDA ops cannot be assumed buildable.

Updated decision after R153: do not attempt Detectron2/MaskDINO installation in the active `yolo-sam-gpu` environment. R154 should be a post-Detectron2-block direction decision: either require an isolated CUDA-toolkit-capable environment before continuing faithful MaskDINO, or pivot to a no-local-CUDA-compile architecture path that can run under the existing PyTorch/timm stack. Any new GPU experiment still needs a fresh smoke/overfit gate and clean-test-v2 must remain final/diagnostic only.

R154 completed the post-Detectron2-block direction at `research-workflow/refine-logs/R154_POST_DETECTRON2_BLOCK_DIRECTION.md`. The decision is to avoid official Detectron2/MaskDINO installation until an isolated CUDA-toolkit-capable environment exists. The next GPU branch is R155, a no-local-CUDA-compile PyTorch/timm SegFormer/DeepLabv3+-style ASPP/context decoder smoke. This is deliberately not another plain U-Net sweep: it should test multi-dilation context aggregation plus shallow high-resolution refinement under the existing boundary/distance supervision and validation gates.

## Compute Policy

- Use one free GPU at a time.
- Current remote target: `/home/shenzeyu/workspace/YOLO_SAM_generic_src`.
- No active GPU run after R129 completion.
- Do not run broad CPU validation grids; keep threshold/mode search narrow.

## R175 Constraint-Metric Pivot

R175 reframes the next phase around the failure modes the user now wants to emphasize: bone-gap adhesion, boundary enhancement, and anatomy-aware component separation. The metrics-only audit at `outputs/analysis/r175_bridge_boundary_claim_metrics.json` shows that R110 remains the strongest completed valid system among available project branches on Dice (`0.917723`), Boundary IoU (`0.251896`), and Precision (`0.913343`). However, false-bridge / separation metrics are incomplete for older patch-arbitrator systems, so the project should not yet claim SOTA superiority on bridge suppression.

The immediate next step is R176, a full mask-synced constraint audit. It must recompute the same bridge/boundary/topology metrics from prediction masks for R090/R100/R110 and selected negative branches. The required claim metrics are Dice, Boundary IoU, separation-band foreground false-positive rate, component-count error, and under-segmentation / false-bridge rate. If R110 already dominates these metrics, the paper/patent story can emphasize a boundary-aware local arbitration advantage. If R110 does not dominate, the next model run should be R177: a boundary-preserving separation arbitrator that edits only train/val-defined bridge-risk regions around the R110 anchor and is judged on both Dice and separation metrics.

Decision gate: no new GPU training before R176, unless the user provides a CUDA-toolkit-capable isolated environment for faithful Mask DINO / Mask2Former or finishes the R168 reviewed CSV path.

R176 completed the mask-synced audit using remote prediction masks for R090/R100/R110/R130/R143/R165. R110 is the best available completed system on Dice (`0.917723`), Boundary IoU (`0.251896`), Boundary F1 (`0.394721`), and separation-band FP (`0.201358`, lower is better). However, R110 is not the best on false-bridge / component consistency: false-bridge is `0.790123` and component-count error is `2.506173`, while R130/R165 reduce component error but lose much more Dice and boundary quality. Therefore the valid claim is not "R110 solves all bone-gap adhesion"; the valid claim is "R110 has the strongest current boundary/region quality, and the remaining bottleneck is under-separation."

Next action is R177: an R110-anchored boundary-preserving separation arbitrator. It should edit only train/val-defined bridge and boundary risk zones, keep R110 unchanged elsewhere, and be judged by a multi-metric gate: Dice must not drop, Boundary IoU/F1 should improve, and false-bridge/component-count error should decrease. Clean-test-v2 remains a locked final evaluation after train/val protocol selection.

## R248 Gate Closure and R251-R253 Claim-Driven Plan (2026-07-10)

### Problem Anchor

- Scope is only `TSRS_RSNA-Epiphysis`; do not mix `TSRS_RSNA-Articular-Surface`.
- The active objective is not to beat nnU-Net on Dice/IoU. Preserve Dice/IoU near ARAA/R110 while improving Boundary IoU, Boundary F1, Surface Dice 2px/5px, HD95, ASSD, gap-region FP rate, component merge rate, and component count MAE.
- The mechanism is reversed topology: preserve/connect seam/background channels between adjacent bones, not foreground-bone connectivity.
- R201 remains the unified evaluation protocol. `clean-test-v2` is locked until the method and thresholds are frozen; it cannot select thresholds, models, or cuts.

### Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
| --- | --- | --- | --- |
| C1: Explicit dense seam/background supervision can distinguish safe background cuts from foreground-eroding cuts at inference without per-image GT. | R244 proves useful cuts exist, while R245/R248 show scalar and local candidate classifiers are unsafe. | Image-grouped original-val evaluation with at least 10 selected images, at least 8 safe-useful, at most 2 risk, Recall delta >= 0, mean cut GT foreground <= 0.05, Boundary IoU/F1 positive, and gap FP negative. | B0, B1, B2 |
| C2: Reducing risk in the generator distribution is more effective than ranking an overwhelmingly unsafe pool. | R246 has 7,590 risk rows versus 971 safe-useful rows. | A seam-constrained generator materially lowers the risk count/rate while retaining useful-candidate coverage on original-val. | B2 |
| Anti-claim: any apparent gain comes from GT-dependent cut selection or clean-test-v2 threshold tuning. | Such a result would be undeployable and invalid under the current protocol. | Training may use inverted train/val GT; inference inputs must be image + anchor/model outputs only. All selection is frozen on original-val before one final clean-test-v2 evaluation. | All |

### R248 Closure

- R248 full grouped CV completed on 96 original-val images and 8,568 candidate rows; no masks were written and clean-test-v2 was not used.
- The original risk-threshold grid was invalid for interpretation: it ended at `0.50`, but OOF risk probabilities ranged from `0.702877` to `0.999949`, so all 49 original configurations selected zero rows and `best=null`.
- R248b reused the same OOF predictions with a reachable grid. Of 80 configurations, 68 were nonempty and 0 passed. The best selected only 4 images with 2 safe-useful and 2 risk cuts; Recall delta was `-0.000015604`.
- Decision: no mask editor and no clean-test-v2. Close the current six-channel local crop dual-head selector as insufficiently clean.

### Experiment Blocks

#### B0: R251-S0 Reverse-Background Supervision Integrity Gate

- Claim tested: inverted bone GT can provide valid seam/background supervision without test-time GT leakage.
- Dataset / split: original `TSRS_RSNA-Epiphysis` train/val only.
- Inputs: grayscale image, non-leaking R110-like anchor proxy, anchor distance/boundary geometry; targets are background labels derived from inverted train/val bone GT only.
- Supervision region: candidate corridors and anchor-interior/ring hard negatives, not the trivial full-image background majority.
- Success criterion: end-to-end smoke completes, train-only overfit shows the seam channel is learnable, and an audit confirms inference code has no GT input.
- Failure interpretation: if the dense channel cannot overfit candidate corridors, do not launch full R251.
- Priority: MUST-RUN.

#### B1: R251 Dense Seam/Background Channel Selector

- Claim tested: dense background evidence is safer than candidate-level crop classification.
- Compared systems: R245 scalar selector, R248 crop selector, R251 dense seam selector.
- Selection data: image-grouped original-val only; no clean-test-v2.
- Decisive metrics: selected images, safe-useful count, risk count, Recall delta, cut GT foreground fraction, Boundary IoU/F1 delta, gap-region FP delta, component count MAE delta.
- Supporting diagnostics: safe-useful ROC-AUC/AP and risk ROC-AUC/AP on candidate pixels/rows; calibration ranges must be reported before threshold grids are defined.
- Success criterion: at least 10 selected images, safe-useful >= 8, risk <= 2, Recall delta >= 0, mean cut GT foreground <= 0.05, Boundary IoU/F1 > 0, gap FP < 0, and component count MAE <= 0.
- Failure interpretation: the candidate pool itself is too contaminated; proceed to B2 and do not add a more complex selector on the same pool.
- Priority: MUST-RUN.

#### B2: R252 Seam-Constrained Candidate Generator

- Claim tested: predicted background-channel geometry can reduce unsafe proposals before ranking.
- Compared systems: R244 generator versus R252 generator with predicted seam-probability support, background-border connectivity, and bone-preservation vetoes.
- Dataset / split: original-val diagnostics only after training on original train; output folder isolated under `outputs/analysis/r252_*`.
- Success criterion: retain useful candidates on at least 50 of the 87 R244 oracle-positive images while reducing risk rows by at least 75%; the final GT-free top-1 gate must meet the B1 safety criteria.
- Failure interpretation: reversed-topology mask editing is not deployable with the current anchor/image evidence; pivot to base-model auxiliary seam-channel training rather than post-hoc cuts.
- Priority: MUST if B1 fails; otherwise NICE-TO-HAVE ablation.

#### B3: R253 Isolated Original-Val Mask Editor and R201 Audit

- Entry gate: only after B1 or B2 passes with frozen weights and thresholds.
- Writes: isolated masks only; never overwrite R110/ARAA masks or raw data.
- Evaluation: full R201-style original-val audit with Dice, IoU, Precision, Recall, Boundary IoU/F1, Surface Dice 2px/5px, HD95, ASSD, gap-region FP rate, component merge rate, and component count MAE.
- Success criterion: Dice/IoU not materially below the R110-like anchor, Recall non-worse, and consistent improvement in the boundary/topology metrics.
- Next step: only after the original-val protocol is frozen may one final `clean-test-v2` evaluation be run; no retuning after seeing it.
- Priority: CONDITIONAL MUST.

### Run Order and Decision Gates

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
| --- | --- | --- | --- | --- | --- |
| M0 | Verify target construction and no leakage | R251-S0 | train-only overfit + inference signature contains no GT | low GPU | class imbalance can yield trivial background output |
| M1 | Test dense seam selector | R251 | strict original-val safety gate above | medium GPU | anchor proxy/domain mismatch |
| M2 | Reduce proposal contamination if needed | R252 | useful coverage retained and risk rows reduced >=75% | CPU/medium GPU | generator may discard rare useful seams |
| M3 | Measure actual mask effect | R253 | complete R201 original-val multi-metric gate | CPU | component fragmentation despite safe cuts |
| M4 | Final locked evaluation | R254 | one frozen clean-test-v2 evaluation only | low | no post-hoc retuning permitted |

### Compute and Data Budget

- One free remote GPU at a time in `/home/shenzeyu/workspace/YOLO_SAM_generic_src` using `/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python`.
- Reuse project launcher conventions in `run_*.sh` and isolate every output directory.
- Do not modify the original dataset layout. No new dataset variant is needed for inverted-GT supervision because the complement target can be generated in memory from original train/val labels.
- Biggest bottleneck: severe safe/risk imbalance inside R244 candidate corridors, not raw classifier capacity alone.
