# Research Brief

## Current execution priority — 2026-09-29

- **Highest Priority Entry Points**:
  - `research-wiki/experiments/ram_so_heatmap_20260929.md`
  - `codex_handoff/RAM_SO_HEATMAP_CONDITIONAL_HANDOFF/00_README_AND_START.md`
- **Active Research Status & Decisions**:
  - **S/O Unified Interface**: Fully validated on RAM with a 737-parameter post-R332-step2 lightweight head (21->32->1), zero-init output layer, signed bound 8. Zero-increment and backward pass verified on synthetic and real cases.
  - **O Hard-Weight Spatial Reweighting**: Concluded as `TRADEOFF / INCONCLUSIVE`. Candidate covered 66 more holdout H0 pixels and 26 val pixels, but full-image AP and overall holdout Dice did not beat base O. Not promoted; no independent long run authorized.
  - **C00 vs CS Completed (2026-09-29)**: Evaluated on holdout84 and val69 under matched evaluator protocol. Holdout Overlap Dice: C00 `0.90556392`, CS `0.90556490`; val C00 `0.87730192`, CS `0.87730057` (val delta `-0.00000136`, descriptive interval spans zero `[-0.00000908, +0.00000709]`). S entry produces localized decisions (21 false positives removed, but 6 true positive members lost) with no macro gain.
  - **Negative Rules / Closed Branches**:
    - **Do NOT launch SO joint training or long runs**: S prior does not yet provide macroscopic positive gain.
    - **Do NOT search O insertion positions**: 2026-09-28 test124 showed both `INPUT_O (0.873789)` and `REFINER_O (0.873814)` underperform `ZERO (0.874588)`.
    - **Do NOT extend H0 final-logit supervision**: 2026-09-24 confirmed overlap Dice dropped by 0.003925pp with 1604 extra FN.
    - **Do NOT launch high-res P1 feature segmentation**: 2026-09-28 showed no stable gain over P0 and 16.5w+ FP.
- **Next Authorized Action (NOT_RUN)**:
  - Directed blind audit of RAM S prior response specifically at real C0 false-connection (gap FP) coordinates to determine signal purity and background false-positive costs before any further model updates.

## Historical execution phase — 2026-09-22

- RAM-W600 is the primary dataset for the current nnU-Net prior project; TSRS is an additional dataset.
- The user reports poorer TSRS label quality and missing consistent color correspondence. Do not assume that colors identify corresponding anatomical members across cases; keep unverified membership/overlap semantics unknown.
- Current work is white-box diagnosis and controlled train/validation experiments, not paper framing. Preserve original GT and evaluation definitions. Do not read test for development.
- Completed the same-tail-LR R332 optimizer-state diagnostic: 100 updates per arm plus two shared-gradient shadow steps. Restoring historical AdamW state did not pass the prespecified performance gate. Keep fresh AdamW with the refiner tail LR as the current continuation reference, not a claim of a globally best final method. Evidence: `outputs/whitebox_adam_20260922/DIAGNOSIS_AND_ACTION.md`; prior LR evidence remains under `outputs/whitebox_20260922`.
- Completed frozen-O remaining-error audit on RAM val69: O>=0.5 covers15.46% of overlap-FN pixels;68.03% of member FN lie in true overlap;40.91% of overlap-member FN are beyond the two-step refiner range when source logits are held fixed. These are diagnostics, not achievable-gain bounds. Evidence: `outputs/ram_o_error_20260922/DIAGNOSIS_AND_ACTION.md`.
- Completed source-gradient diagnosis and one matched100-update pair: unselected hard FN receive weaker direct source gradients; adding targeted source BCE improves overlap DSC by0.000199335 but lowers overlap NSD by0.000054061 and does not recover the initial71,337 hardest val members. It fails the preset gate and is not promoted. Evidence: `outputs/ram_source_probe_20260922/DIAGNOSIS_AND_ACTION.md`; all historical anchors remain preserved.
- Completed feature/gradient audit: most newFP follows backbone changes; sampled source shifts are about98% attributable to decoder features, not head bias. NewFP starts near final decision boundary while hardFN remains far below it. Local FN/FP gradients conflict moderately, not enough evidence to declare the network incapable. Evidence: `outputs/ram_feature_probe_20260922/DIAGNOSIS_AND_ACTION.md`.
- Numeric protocol: explicitly fix torch CPU normalization threads to64 when matching these pilots. Changing to2 changed image normalization tensor hashes and tiny NSD values; matched64-thread recomputation exactly reproduced original NSD. Preserve both diagnostic versions; do not overwrite old scores.
- Completed local GT-negative protection verification: 6 independent shadow updates and one100-update variant matched to reusable positive-only100. Same100batch hashes/member selections, no AMP skips; teacher frozen and excluded from optimizer. Guard reduces FP by2441 but increases FN by2404; hardFN recovery remains0. OverallNSD improves vs positive-only, but overlapNSD falls by0.000055806 and the preset gate fails. Do not promote or automatically extend this guard. Evidence: `outputs/ram_negative_guard_20260922/DIAGNOSIS_AND_ACTION.md`.
- Completed identity-context investigation under `outputs/ram_identity_causal_20260922`: train-only association and exactfull69zero-branch parity passed, but samecapacity850-update pair FAILED material-gain gate. Canonical vs scrambled OverlapDSC delta=0.0000273083 (0.002731pp); descriptive case-bootstrap interval includeszero. Both worse than commonfresh100anchor. Branch is used (673 maskchanges whenoff), buthardFN71335 recovered0. Do not promote or continue tuning this identity adapter. Pair AMP1024 completed1700updates; failed/replay322updates also preserved; conservative total63min<2GPUh. OriginalAMP65536 failure diagnosed through approximate replay; never call it exact recovery.
- Next NOT_RUN priority: controlled train-case correctability diagnosis of high-confidence source errors under current versus deeper-feature trainability, before considering another validation pair. Frozen representations, loss paths, and exposure remain alternative explanations; do not claim a unique cause or guarantee unfreezing gains. A new protocol must be locked separately. No queued continuation or extra run is authorized by this note.
- The older project framing below is historical context and does not override this priority or authorize its old queues/test usage.

## Problem Statement

Improve the current `YOLO+SAM` epiphysis segmentation line until it clearly matches or exceeds the ARAA benchmark under the user's aligned environment, with `clean-test-v2` as the primary decision slice and the original test split kept as a secondary control slice.

## Current Project Position

- The project is no longer in an early exploration phase.
- A strong internal refiner line already exists: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`.
- A strong external anchor also exists: ARAA best `clean-test-v2` checkpoint `98.pth`.
- The remaining gap is small enough that future work must be more disciplined and anatomy-targeted, not just another generic refinement variant.

## Locked Benchmark Context

### External target

- **Method**: ARAA
- **Repo on server**: `/home/shenzeyu/workspace/ARAA-Net`
- **Primary score to beat**: Dice `0.911766` on `TSRS_RSNA-Epiphysis_clean_test_v2`
- **Key profile**: slightly stronger recall and boundary completeness than the current best refiner

### Current internal anchor

- **Method**: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`
- **Primary comparison slice**: `TSRS_RSNA-Epiphysis_clean_test_v2`
- **Recorded score**: Dice `0.910029`
- **Current interpretation**: very close to ARAA, but still slightly too conservative

## Research Goal

Push the current refiner family from "nearly tied with ARAA" to "convincingly competitive or better than ARAA", especially on the cases that currently determine failure:

- thin epiphysis under-segmentation
- narrow inter-bone gap collapse
- overly smooth masks that look cleaner but lose anatomy

## Main Bottleneck

The main remaining difficulty is no longer coarse localization. It is the trade-off between:

- preserving thin positive bone structures
- keeping adjacent bone gaps open
- improving boundary quality without erasing small anatomy

This is why `2982`, `1475`, `3040`, and `1433` remain important qualitative gate cases.

## Evaluation Policy

### Primary slice

- `TSRS_RSNA-Epiphysis_clean_test_v2`
- use this to judge whether a method really beats the current ARAA target under cleaner evaluation conditions

### Secondary slice

- original `TSRS_RSNA-Epiphysis` test
- use this to verify that gains are not only artifacts of evaluation cleanup

### Reporting rule

Every promoted result should report both slices when possible. A result should not be called the new mainline if it only looks good on the cleaned slice while collapsing on the original split.

## Method Direction

Future variants should stay grounded in the current structure-aware refiner line and emphasize:

- thin-structure preservation
- explicit gap protection between neighboring bones
- prediction-aware or structure-aware boundary correction

Methods that merely smooth contours or rely on blunt post-processing are not the main direction unless they also preserve anatomy on hard cases.

## Dataset and Experiment Discipline

- Preserve the original dataset layout.
- Keep `TSRS_RSNA-Epiphysis` and `TSRS_RSNA-Articular-Surface` fully separated.
- Any filtering or suspicious-label removal must create a new manifest and a new isolated dataset variant.
- Filtered-data gains must be reported separately from pure architecture gains.
- Every new experiment must use isolated output folders and explicit launcher names.

## Success Criteria

A new method direction is worth promoting only if it satisfies most or all of the following:

1. Beats the current strongest internal refiner on `clean-test-v2`.
2. Reaches or exceeds the ARAA target Dice `0.911766`, or comes with a very strong qualitative and boundary-quality case if the score is only marginally below.
3. Does not visibly worsen the hard cases centered on thin epiphysis and bone-gap preservation.
4. Keeps the claim chain honest across `clean-test-v2`, original test, and any filtered-data control runs.

## Immediate Next Use In ARIS

This brief is now aligned to support:

- `/experiment-bridge` for the next ARAA-targeted experiment campaign
- structured bad-case-driven method selection
- future `/auto-review-loop` runs that judge whether a candidate should become the new project mainline

## Non-Goals

- Replacing the whole project with a different framework
- Mixing cleaned-data and default-data conclusions into one result line
- Promoting visually smoother masks that actually damage thin anatomy
- Treating small metric movements as meaningful without bad-case validation
