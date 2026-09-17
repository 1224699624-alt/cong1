# R211-F1 Learned Acceptance Gate Progress

Date: 2026-07-07

## Purpose

Convert R210-F1 from a GT-gated feasibility method into a deployable-style GT-free acceptance method.

R211-F1 trains a candidate acceptor on original train/val only:

- R210-F1 neck candidates are generated from R110 anchor masks.
- Candidate features use only inference-time information.
- GT is used only to create train/val labels and to evaluate validation predictions.
- At inference, the learned acceptor uses only candidate/image/anchor features.
- `clean-test-v2` is not used.

## Implementation

Added:

- `scripts/run_r211_f1_learned_acceptance_gate.py`

Key implementation details:

- Candidate features:
  - component area,
  - cut area,
  - cut/component fraction,
  - split-gain effect on the anchor,
  - piece count,
  - candidate bbox size,
  - slenderness/fill,
  - image-gradient ratio,
  - distance-transform ratio,
  - anchor component count,
  - anchor area,
  - candidate rank.
- Classifier:
  - default `logreg` with balanced class weight,
  - optional `hgb`.
- Candidate labels are GT-derived only on train/val.
- Validation prediction uses the classifier probability and a threshold grid.
- Final per-image val metrics use R201 metrics.

Performance fix:

- Avoided building a full split-level GT cache at startup.
- Candidate labeling now builds per-image GT cache and uses lightweight candidate metrics by default.
- Full R201 surface metrics are still used in final per-image validation evaluation.

## Remote Quick Smoke

Remote workspace:

- `/home/shenzeyu/workspace/YOLO_SAM_generic_src`

Command shape:

```bash
/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin/python \
  scripts/run_r211_f1_learned_acceptance_gate.py \
  --limit-train 4 \
  --limit-val 4 \
  --max-candidates-generated 4 \
  --max-components-per-image 2 \
  --threshold-grid 0.3,0.5 \
  --output-exp r211_f1_learned_acceptance_gate_remote_quick4
```

Artifacts:

- `outputs/analysis/r211_f1_remote_quick4_train_candidates.csv`
- `outputs/analysis/r211_f1_remote_quick4_val_candidates.csv`
- `outputs/analysis/r211_f1_remote_quick4_val_summary.json`
- `outputs/analysis/r211_f1_remote_quick4_val_per_image.csv`
- `outputs/bridge_logs/r211_f1_remote_quick4.log`

Quick4 result:

- train candidates: `8`
- train positives: `5`
- val candidates: `8`
- val positives: `3`
- best threshold: `0.3`
- accepted rate: `0.5`
- Dice delta: `-0.000150`
- IoU delta: `-0.000227`
- Recall delta: `-0.000365`
- gap FP delta: `-0.000209`
- Boundary IoU delta: `-0.000064`
- Surface Dice 2px delta: `-0.000409`
- component count MAE delta: `0.0`
- gate pass: `true`

Interpretation:

The quick smoke proves the R211-F1 pipeline works end to end on the server. This is not a model claim because the sample size is only `4` val images.

## Active Remote Run

Started a larger train/val smoke:

```bash
/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin/python \
  scripts/run_r211_f1_learned_acceptance_gate.py \
  --limit-train 64 \
  --limit-val 32 \
  --max-candidates-generated 8 \
  --max-components-per-image 3 \
  --threshold-grid 0.3,0.4,0.5,0.6,0.7 \
  --output-exp r211_f1_learned_acceptance_gate_remote_val32
```

Remote PID:

- `1579798`

Log:

- `outputs/bridge_logs/r211_f1_remote_val32.log`

Expected artifacts:

- `outputs/analysis/r211_f1_remote_val32_train_candidates.csv`
- `outputs/analysis/r211_f1_remote_val32_val_candidates.csv`
- `outputs/analysis/r211_f1_remote_val32_val_summary.json`
- `outputs/analysis/r211_f1_remote_val32_val_per_image.csv`

Status at last check:

- process alive,
- train64 candidate generation completed,
- val32 candidate generation completed,
- threshold evaluation started and reached `eval/val/t0.30`,
- no runtime error observed,
- no final summary yet.

Update note:

- The run is slower than quick4 because some val cases, especially later hard cases, take tens of seconds during candidate generation and full R201 validation evaluation.
- Do not kill this run unless it stalls or errors; it has already moved past candidate generation and is producing the validation threshold grid.
- A full train875/full-val run is probably too slow for the current unoptimized candidate code. If val32 is promising, the next practical step should be full original-val evaluation with the learned gate, but classifier training may use a representative train subset rather than all 875 images.

Second monitor update:

- PID `1579798` remains alive and error-free.
- `t0.30` threshold evaluation completed over `32/32` validation images.
- `t0.40` threshold evaluation is in progress and was observed around `16/32`.
- No final summary JSON exists yet.
- Engineering note: threshold evaluation recomputes candidate generation and full metrics for each threshold. If R211-F1 survives this smoke, the next script improvement should cache per-image candidate scores once and sweep thresholds over cached candidates to avoid repeated expensive evaluation.

Third monitor update:

- PID `1579798` remains alive and error-free.
- Threshold `t0.30` completed.
- Threshold `t0.40` completed.
- Threshold `t0.50` was observed around `29/32`.
- Final summary JSON is still absent.
- The run is still making progress, but the repeated threshold-evaluation design is now the main bottleneck. If this run does not finish soon, prefer implementing cached threshold evaluation over launching any larger R211-F1 run.

Fourth monitor / engineering update:

- PID `1579798` remains alive and error-free.
- Thresholds `t0.30`, `t0.40`, `t0.50`, and `t0.60` completed.
- Threshold `t0.70` is the final threshold and was observed in progress on validation images.
- Final summary JSON is still absent at this checkpoint.
- Implemented cached threshold evaluation in `scripts/run_r211_f1_learned_acceptance_gate.py`: each validation image now generates and scores R210-F1 neck candidates once, then sweeps the threshold grid over cached candidate scores.
- Local equivalence smoke passed on one synced validation case across thresholds `0.3`, `0.5`, and `0.7`: cached and uncached application produced identical predictions and matching key deltas.
- Synced the optimized script to the remote workspace and verified remote `py_compile`.
- The active `val32` process was not interrupted; it continues with the old already-loaded code. Use the cached script for any next `val32` rerun or full original-val evaluation.

## Val32 Result

Artifacts synced locally:

- `outputs/analysis/r211_f1_remote_val32_train_candidates.csv`
- `outputs/analysis/r211_f1_remote_val32_val_candidates.csv`
- `outputs/analysis/r211_f1_remote_val32_val_summary.json`
- `outputs/analysis/r211_f1_remote_val32_val_per_image.csv`
- `outputs/bridge_logs/r211_f1_remote_val32.log`

Result:

- num evaluated: `32`
- train candidates: `188`
- train positives: `100`
- val candidates: `85`
- val positives: `45`
- best threshold: `0.7`
- val gate pass: `false`

Best-threshold mean deltas vs R110 val anchor:

- accepted image rate: `0.125`
- Dice: `-0.000028`
- IoU: `-0.000053`
- Recall: `-0.000129`
- Boundary IoU: `-0.000168`
- Boundary F1: `-0.000235`
- Surface Dice 2px: `-0.000397`
- Surface Dice 5px: `+0.000083`
- HD95 px: `-0.002834`
- ASSD px: `-0.000103`
- gap-region FP rate: `-0.000145`
- component merge rate: `0.000000`
- component count MAE: `+0.343750`
- instance merge count: `0.000000`

Interpretation:

R211-F1 does not pass. It can reduce gap-region FP slightly, but the learned acceptor still admits cuts that increase component-count error. The most obvious per-image risk is `1585.png`, where the best-threshold output improves gap FP and Dice slightly but increases component count MAE by `+10`. This violates the project requirement that adhesion reduction must not be bought through fragmentation or over-cutting.

## R211-F2 Component-Guard Follow-Up

Implemented a GT-free component-count safety guard in `scripts/run_r211_f1_learned_acceptance_gate.py`.

The guard uses prediction topology only:

- `--max-pred-component-increase`
- `--max-step-component-increase`

Default values are intentionally permissive, preserving R211-F1 behavior when the new options are not passed. Local smoke confirmed cached/default behavior remains equivalent on a synced validation case.

Started R211-F2 on the remote server:

```bash
/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin/python \
  scripts/run_r211_f1_learned_acceptance_gate.py \
  --limit-train 64 \
  --limit-val 32 \
  --max-candidates-generated 8 \
  --max-components-per-image 3 \
  --max-pred-component-increase 0 \
  --max-step-component-increase 0 \
  --threshold-grid 0.3,0.4,0.5,0.6,0.7 \
  --output-exp r211_f2_component_guard_remote_val32
```

Remote PID:

- `1590009`

Expected artifacts:

- `outputs/analysis/r211_f2_remote_val32_train_candidates.csv`
- `outputs/analysis/r211_f2_remote_val32_val_candidates.csv`
- `outputs/analysis/r211_f2_remote_val32_val_summary.json`
- `outputs/analysis/r211_f2_remote_val32_val_per_image.csv`
- `outputs/bridge_logs/r211_f2_remote_val32.log`

This is still train/val only. `clean-test-v2` remains locked.

## R211-F2 Val32 Result

Artifacts synced locally:

- `outputs/analysis/r211_f2_remote_val32_train_candidates.csv`
- `outputs/analysis/r211_f2_remote_val32_val_candidates.csv`
- `outputs/analysis/r211_f2_remote_val32_val_summary.json`
- `outputs/analysis/r211_f2_remote_val32_val_per_image.csv`
- `outputs/bridge_logs/r211_f2_remote_val32.log`

Result:

- num evaluated: `32`
- best threshold: `0.6`
- val gate pass: `true`

Best-threshold mean deltas vs R110 val anchor:

- accepted image rate: `0.187500`
- Dice: `-0.000028`
- IoU: `-0.000049`
- Recall: `-0.000094`
- Boundary IoU: `+0.000025`
- Boundary F1: `+0.000033`
- Surface Dice 2px: `-0.000206`
- Surface Dice 5px: `+0.000001`
- HD95 px: `0.000000`
- ASSD px: `+0.000200`
- gap-region FP rate: `-0.000112`
- component merge rate: `0.000000`
- component count MAE: `-0.031250`
- instance merge count: `0.000000`
- predicted component count: `0.000000`

Interpretation:

R211-F2 passes the `val32` gate, but the effect size is small. The component-count guard fixes the main R211-F1 failure mode: high-risk cuts such as `1585.png` and `1452.png` are rejected, while safer edits such as `15040.png` and `14041.png` remain possible. This is directionally aligned with the project goal because gap FP and boundary metrics improve slightly without increasing predicted component count or component-count MAE.

This is not sufficient for final claims. The next required step is full original-val evaluation with the same GT-free guard before any clean-test-v2 action.

## R211-F2 Full Original-Val Launch

Started full original-val evaluation using the same F2 component guard:

```bash
/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin/python \
  scripts/run_r211_f1_learned_acceptance_gate.py \
  --limit-train 64 \
  --limit-val 0 \
  --max-candidates-generated 8 \
  --max-components-per-image 3 \
  --max-pred-component-increase 0 \
  --max-step-component-increase 0 \
  --threshold-grid 0.3,0.4,0.5,0.6,0.7 \
  --output-exp r211_f2_component_guard_remote_fullval
```

Remote PID:

- `1593604`

Expected artifacts:

- `outputs/analysis/r211_f2_remote_fullval_train_candidates.csv`
- `outputs/analysis/r211_f2_remote_fullval_val_candidates.csv`
- `outputs/analysis/r211_f2_remote_fullval_val_summary.json`
- `outputs/analysis/r211_f2_remote_fullval_val_per_image.csv`
- `outputs/bridge_logs/r211_f2_remote_fullval.log`

This is still original train/val only. `clean-test-v2` remains locked.

## Decision

Do not use quick4 as evidence for clean-test-v2.

Next ARIS step:

1. Monitor `r211_f1_remote_val32`.
2. If val32 passes with nonzero anatomy improvement and safe Dice/Recall/Boundary deltas, run full original val.
3. If val32 fails, inspect candidate labels and classifier threshold behavior before redesigning.
4. Keep `clean-test-v2` locked until a full original-val GT-free gate passes and hard-case visual audit is complete.

## R211-F2 Full Original-Val Result

Artifacts synced locally:

- `outputs/analysis/r211_f2_remote_fullval_train_candidates.csv`
- `outputs/analysis/r211_f2_remote_fullval_val_candidates.csv`
- `outputs/analysis/r211_f2_remote_fullval_val_summary.json`
- `outputs/analysis/r211_f2_remote_fullval_val_per_image.csv`
- `outputs/bridge_logs/r211_f2_remote_fullval.log`

Derived audit artifacts:

- `outputs/analysis/r211_f2_remote_fullval_case_summary.json`
- `outputs/analysis/r211_f2_remote_fullval_visual_candidates.csv`

Result:

- num evaluated: `96`
- best threshold: `0.6`
- accepted images: `33/96`
- accepted image rate: `0.343750`
- val gate pass: `true`

Best-threshold mean deltas vs R110 val anchor:

- Dice: `-0.000030`
- IoU: `-0.000055`
- Recall: `-0.000178`
- Boundary IoU: `-0.000070`
- Boundary F1: `-0.000099`
- Surface Dice 2px: `-0.000198`
- Surface Dice 5px: `+0.000098`
- HD95 px: `-0.006130`
- ASSD px: `-0.000467`
- gap-region FP rate: `-0.000292`
- component merge rate: `0.000000`
- component count MAE: `-0.010417`
- predicted component count: `0.000000`

Interpretation:

R211-F2 passes full original-val as a topology-safe GT-free acceptance gate, but the effect size is small and the boundary overlap metrics are slightly negative. The strongest positive evidence is that the component guard keeps predicted component count unchanged while slightly improving gap-region FP, HD95, ASSD, and component-count MAE. This supports the component-preserving direction, but it is not yet strong enough to justify a clean-test-v2 final application as a new mainline.

Decision:

- Do not apply R211-F2 to clean-test-v2 yet.
- Use the generated visual candidate list for hard-case inspection.
- Keep the R211-F2 component guard as a required safety mechanism for the next version.
- Next method iteration should increase boundary-positive / surface-positive acceptance pressure while preserving the GT-free component guard, because the current gate improves adhesion diagnostics only weakly and slightly hurts Boundary IoU/F1.
