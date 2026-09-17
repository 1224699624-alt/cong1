# R212 Boundary-Positive Acceptance Progress

Date: 2026-07-07

## Purpose

R211-F2 showed that GT-free component guarding can prevent severe fragmentation, but its full-val gains were too small and Boundary IoU/F1 were slightly negative. R212 tests a stricter candidate-labeling direction: accept only candidates that are not merely gap-reducing, but also boundary/surface-positive under R201 metrics.

This remains original train/val only. `clean-test-v2` is not used.

## Implementation

Extended `scripts/run_r211_f1_learned_acceptance_gate.py` with backward-compatible R212 options:

- `--label-mode boundary_positive`
- `--label-min-boundary-gain`
- `--label-min-surface2-gain`
- `--label-max-hd95-delta`
- `--label-max-assd-delta`
- `--strict-boundary-gate`

Default behavior remains the earlier R211 label/gate behavior unless these options are enabled.

R212-F0 uses:

- full candidate labels via R201 metrics;
- GT-derived labels on train/val only;
- inference-time candidate features only;
- component guard:
  - `--max-pred-component-increase 0`
  - `--max-step-component-increase 0`
- strict validation gate requiring non-negative Boundary IoU/F1 and Surface Dice 2px mean deltas.

## R212-F0 LogReg Val32 Smoke

Remote command used train64/val32 with `classifier=logreg`.

Artifacts:

- `outputs/analysis/r212_f0_remote_val32_train_candidates.csv`
- `outputs/analysis/r212_f0_remote_val32_val_candidates.csv`
- `outputs/analysis/r212_f0_remote_val32_val_summary.json`
- `outputs/analysis/r212_f0_remote_val32_val_per_image.csv`
- `outputs/bridge_logs/r212_f0_remote_val32.log`

Candidate-label sanity:

- train candidates: `188`
- train positives: `65`
- train positive rate: `0.345745`
- val candidates: `85`
- val positives: `28`
- val positive rate: `0.329412`

Positive candidate means are well aligned:

- train positives: Dice `+0.000240`, Boundary IoU `+0.001568`, Boundary F1 `+0.001876`, Surface Dice 2px `+0.001248`, gap FP `-0.001318`, component count MAE `-0.076923`
- val positives: Dice `+0.000175`, Boundary IoU `+0.001443`, Boundary F1 `+0.001828`, Surface Dice 2px `+0.001360`, gap FP `-0.001380`, component count MAE `-0.035714`

This confirms that the boundary-positive label definition is not empty and separates useful candidates from harmful ones.

Best threshold: `0.7`.

Mean deltas vs R110 anchor on val32:

- accepted image rate: `0.125000`
- Dice: `-0.000027`
- IoU: `-0.000046`
- Recall: `-0.000087`
- Boundary IoU: `-0.000065`
- Boundary F1: `-0.000073`
- Surface Dice 2px: `-0.000055`
- Surface Dice 5px: `-0.000001`
- HD95 px: `0.000000`
- ASSD px: `+0.000242`
- gap-region FP rate: `-0.000086`
- component merge rate: `0.000000`
- component count MAE: `-0.031250`
- predicted component count: `0.000000`

Gate result: `false`.

## Failure Diagnosis

The label definition worked, but the linear logistic acceptor did not select candidates cleanly enough.

Accepted cases at the best threshold:

- Good:
  - `15067.png`: Boundary IoU `+0.001604`, Surface Dice 2px `+0.001504`, gap FP `-0.001545`
  - `10520.png`: Boundary IoU `+0.000430`, Surface Dice 2px `+0.000135`, gap FP `-0.001054`
- Bad:
  - `12627.png`: Boundary IoU `-0.002318`, no gap gain
  - `15040.png`: Boundary IoU `-0.001790`, Surface Dice 2px `-0.003310`, component count MAE `-1.0`

Interpretation:

R212-F0 is not a passed model, but it provides a positive mechanism signal: boundary-positive candidates exist and are separable in GT-derived labels. The failure is likely the linear acceptor/ranking, not the candidate-label concept.

## Decision

- Do not run full-val or clean-test from R212-F0.
- Run R212-F1 with the same labels, same component guard, and same strict gate, but switch the acceptor to `hgb` to test whether a nonlinear classifier avoids the bad accepted cuts.
- If HGB still accepts boundary-negative cuts, inspect feature importance / add explicit prediction-time boundary-risk features before another full-val run.

## R212-F1 HGB Val32 Smoke

Remote command repeated the R212-F0 setup but used `--classifier hgb`.

Artifacts:

- `outputs/analysis/r212_f1_remote_val32_train_candidates.csv`
- `outputs/analysis/r212_f1_remote_val32_val_candidates.csv`
- `outputs/analysis/r212_f1_remote_val32_val_summary.json`
- `outputs/analysis/r212_f1_remote_val32_val_per_image.csv`
- `outputs/bridge_logs/r212_f1_remote_val32.log`

Candidate-label counts match R212-F0:

- train candidates: `188`
- train positives: `65`
- train positive rate: `0.345745`
- val candidates: `85`
- val positives: `28`
- val positive rate: `0.329412`

Best threshold: `0.7`.

Mean deltas vs R110 anchor on val32:

- accepted image rate: `0.125000`
- Dice: `-0.000010`
- IoU: `-0.000017`
- Recall: `-0.000046`
- Boundary IoU: `+0.000011`
- Boundary F1: `+0.000011`
- Surface Dice 2px: `+0.000007`
- Surface Dice 5px: `-0.000018`
- HD95 px: `0.000000`
- ASSD px: `+0.000016`
- gap-region FP rate: `-0.000070`
- component merge rate: `0.000000`
- component count MAE: `0.000000`
- predicted component count: `0.000000`

Gate result: `true`.

Accepted cases:

- Good:
  - `15067.png`: Boundary IoU `+0.001604`, Surface Dice 2px `+0.001504`, gap FP `-0.001545`
  - `15446.png`: Boundary IoU `+0.000238`, Surface Dice 2px `+0.000621`, gap FP `-0.000705`
- Risk:
  - `13867.png`: Boundary IoU `-0.000251`, Surface Dice 2px `-0.000559`, no gap gain
  - `15040.png`: Boundary IoU `-0.001251`, Surface Dice 2px `-0.001331`, no gap gain

## R212-F1 Interpretation

HGB improves over the logistic acceptor and passes the strict val32 gate. This confirms that nonlinear candidate acceptance is a better fit for the boundary-positive labels.

However, the effect is still very small and the accepted set still contains boundary-negative/no-gap-gain cases. The mean gate passes because two positive cases compensate for two weaker cases, not because every accepted edit is clean. This is not yet strong enough for full original-val expansion or clean-test-v2.

## Updated Decision

Verdict: `small_positive_smoke_not_promoted`.

- Do not apply R212-F1 to clean-test-v2.
- Do not yet run full original-val as a promoted candidate.
- Continue to R212-F2 by adding prediction-time risk controls before acceptance:
  - require a candidate probability margin or higher threshold grid above `0.7`;
  - add explicit inference-time boundary-risk features, such as candidate overlap with high-gradient boundary support, cut slenderness/size interactions, and local anchor-boundary disruption;
  - reject accepted edits with no predicted gap/neck confidence gain once such features are available;
  - keep `max_pred_component_increase=0` and `max_step_component_increase=0`.

Promotion requirement for the next smoke:

- Boundary IoU/F1 and Surface Dice 2px must remain non-negative;
- gap FP must improve more than R212-F1;
- accepted cases should not include obvious boundary-negative/no-gap-gain edits like `13867.png` and `15040.png`.

## R212-F2 Boundary-Risk Filter Launch

R212-F2 adds backward-compatible inference-time acceptance filters to
`scripts/run_r211_f1_learned_acceptance_gate.py`. Defaults preserve prior
R211/R212 behavior; the new filters are enabled only by explicit flags:

- `--max-accept-candidate-rank`
- `--min-accept-gradient-ratio`
- `--min-accept-dist-ratio`
- `--min-accept-slenderness`
- `--max-accept-fill`

The per-image CSV now also records `accepted_candidate_summary`, which lists
the accepted candidate rank, probability, cut size, split features, and
gradient/distance support. This is needed because R212-F1 passed the mean gate
while still accepting boundary-negative candidates.

Local validation:

- `python -m py_compile scripts/run_r211_f1_learned_acceptance_gate.py` passed.
- A local schema smoke on synced val masks completed and wrote the new field;
  it is pipeline evidence only, not a model result.

Remote val32 smoke launched:

- screen: `r212_f2_boundary_filter_val32`
- log: `outputs/bridge_logs/r212_f2_boundary_filter_val32.log`
- summary target: `outputs/analysis/r212_f2_remote_val32_val_summary.json`
- per-image target: `outputs/analysis/r212_f2_remote_val32_val_per_image.csv`
- candidate CSVs:
  - `outputs/analysis/r212_f2_remote_val32_train_candidates.csv`
  - `outputs/analysis/r212_f2_remote_val32_val_candidates.csv`
- checkpoint:
  `outputs/r211_f1_learned_acceptance_gate/r212_f2_boundary_filter_hgb_remote_val32.joblib`

Key command choices:

- original train/val only; clean-test-v2 not used;
- `classifier=hgb`;
- boundary-positive labels with full R201 candidate labels;
- component guard retained:
  `--max-pred-component-increase 0`,
  `--max-step-component-increase 0`;
- threshold grid extended to `0.70,0.75,0.80,0.85,0.90,0.95`;
- inference filter:
  rank `<=2`, gradient ratio `>=2.0`, distance ratio `>=0.12`,
  slenderness `>=4.0`, fill `<=0.80`.

Expected audit after completion:

- compare accepted cases against R212-F1, especially `13867.png` and
  `15040.png`;
- require non-negative Boundary IoU/F1 and Surface Dice 2px;
- require gap FP improvement stronger than R212-F1 if accepted rate remains
  nonzero;
- do not promote to full-val or clean-test unless the accepted candidate audit
  no longer contains obvious boundary-negative/no-gap-gain edits.

## R212-F2 Interim Monitor

Remote screen `r212_f2_boundary_filter_val32` is still running normally and
has not produced the final summary yet. At the latest check it was still in
train64 candidate generation; no traceback was present in
`outputs/bridge_logs/r212_f2_boundary_filter_val32.log`.

An offline replay using the already-synced R212-F1 train/val candidate CSVs
was run only as a parameter sanity check, not as model evidence. It retrained
the same HGB acceptor on `r212_f1_remote_val32_train_candidates.csv` and
applied the R212-F2 filter to the existing candidate rows. This replay suggests
that:

- threshold `0.80` would keep `15067.png` and `15446.png`, both positive
  boundary/gap candidates;
- threshold `0.80` would remove the known R212-F1 risk accepts `13867.png`
  and `15040.png`;
- threshold `0.85` would become more conservative and keep only
  `15067.png`;
- thresholds `0.90` and above may become no-op.

This is only approximate because the real remote evaluation recomputes the
sequential acceptance path and full R201 metrics. The next decision must wait
for `r212_f2_remote_val32_val_summary.json` and
`r212_f2_remote_val32_val_per_image.csv`.

## R212-F2 Val32 Result

Remote R212-F2 completed on original train/val only. `clean-test-v2` was not
used.

Artifacts synced locally:

- `outputs/analysis/r212_f2_remote_val32_train_candidates.csv`
- `outputs/analysis/r212_f2_remote_val32_val_candidates.csv`
- `outputs/analysis/r212_f2_remote_val32_val_summary.json`
- `outputs/analysis/r212_f2_remote_val32_val_per_image.csv`
- `outputs/bridge_logs/r212_f2_boundary_filter_val32.log`
- `outputs/r211_f1_learned_acceptance_gate/r212_f2_boundary_filter_hgb_remote_val32.joblib`

Candidate labels:

- train candidates: `304`
- train positives: `96`
- train positive rate: `0.315789`
- val candidates: `137`
- val positives: `39`
- val positive rate: `0.284672`

Best threshold: `0.75`.

Mean deltas vs R110 anchor on val32:

- accepted image rate: `0.031250`
- Dice: `-0.0000003`
- IoU: `-0.0000005`
- Recall: `-0.0000094`
- Boundary IoU: `+0.0000074`
- Boundary F1: `+0.0000100`
- Surface Dice 2px: `+0.0000194`
- Surface Dice 5px: `-0.0000070`
- HD95 px: `0.000000`
- ASSD px: `-0.0000174`
- gap-region FP rate: `-0.0000220`
- component merge rate: `0.000000`
- component count MAE: `0.000000`
- predicted component count delta: `0.000000`
- instance merge count delta: `0.000000`

Gate result: `true`.

Accepted case:

- `15446.png`: accepted candidate rank `1`, probability `0.885016`,
  Boundary IoU `+0.000238`, Boundary F1 `+0.000321`,
  Surface Dice 2px `+0.000621`, gap FP `-0.000705`,
  Dice `-0.000009`, Recall `-0.000300`, component count MAE `0.0`.

Known R212-F1 risk cases:

- `13867.png`: no longer accepted.
- `15040.png`: no longer accepted.

Missed useful case:

- `15067.png`: no longer accepted despite a positive candidate:
  candidate rank `2`, probability `0.678770`, Boundary IoU `+0.001604`,
  Surface Dice 2px `+0.001504`, gap FP `-0.001545`, component count MAE
  `0.0`.

## R212-F2 Interpretation

Verdict: `technical_pass_not_promoted`.

R212-F2 achieved the narrow safety goal: it removed the obvious bad accepts
from R212-F1 while keeping mean boundary/surface/gap deltas non-negative.
However, it became too conservative:

- only `1/32` validation images were edited;
- mean gains are an order of magnitude too small to justify full-val or
  clean-test promotion;
- the filter missed `15067.png`, one of the clearest positive examples;
- the probability ranking still assigns high scores to unsafe candidates, for
  example `15040.png` candidate 1 has probability `0.732052` despite negative
  boundary/surface deltas, so thresholding alone cannot solve the acceptor.

Decision:

- Do not apply R212-F2 to clean-test-v2.
- Do not promote R212-F2 to full original-val as a model candidate.
- Continue to R212-F3 by improving the acceptor signal rather than simply
  tightening filters.

Recommended R212-F3 direction:

1. Keep the component guard `0/0` and accepted-candidate audit.
2. Add an explicit train-time negative emphasis for high-probability unsafe
   cases such as `15040.png` candidate 1 and `13480.png` candidates 1/2.
3. Add or derive inference-time features that distinguish positive
   boundary-supported cuts from harmful cuts:
   - candidate boundary-contact fraction;
   - cut-to-anchor-boundary overlap;
   - local foreground-removal recall-risk proxy;
   - whether the cut lies in a high-curvature/neck region rather than a broad
     object side;
   - stricter handling of `split_gain == 0` candidates with weak gap evidence.
4. Retest on val32 first. Promotion requires nonzero accepted rate above F2,
   no `13867.png`/`15040.png` style bad accepts, and stronger gap/boundary
   mean deltas than R212-F2.

## R212-F3 Shape-Feature Acceptor Launch

R212-F3 was launched to address the main R212-F2 failure: the HGB acceptor
still gave high probability to unsafe cuts such as `15040.png` candidate 1,
while the F2 threshold/filter became too conservative and missed useful cuts
such as `15067.png` candidate 2.

Code change:

- Extended `scripts/run_r211_f1_learned_acceptance_gate.py` with additional
  GT-free candidate shape/contact features:
  - `component_edge_contact_frac`
  - `cut_boundary_contact_frac`
  - `cut_exposed_boundary_frac`
  - `cut_neighbor_fg_frac`
  - `cut_centroid_dist_norm`
- The new features are included in `FEATURE_KEYS`, so they are used by the HGB
  acceptor during training and inference.
- The existing F2 inference filters remain available and default-off.

Validation before launch:

- Local `python -m py_compile scripts/run_r211_f1_learned_acceptance_gate.py`
  passed.
- Local schema smoke wrote the new feature columns into
  `outputs/analysis/r212_f3_local_schema_smoke_train_candidates.csv`.
- Remote py_compile passed after syncing the script.

Remote val32 smoke launched:

- screen: `r212_f3_shape_features_val32`
- log: `outputs/bridge_logs/r212_f3_shape_features_val32.log`
- summary target: `outputs/analysis/r212_f3_remote_val32_val_summary.json`
- per-image target: `outputs/analysis/r212_f3_remote_val32_val_per_image.csv`
- candidate CSVs:
  - `outputs/analysis/r212_f3_remote_val32_train_candidates.csv`
  - `outputs/analysis/r212_f3_remote_val32_val_candidates.csv`
- checkpoint:
  `outputs/r211_f1_learned_acceptance_gate/r212_f3_shape_features_hgb_remote_val32.joblib`

Key command choices:

- original train/val only; clean-test-v2 not used;
- `classifier=hgb`;
- full R201 candidate labels;
- boundary-positive label mode;
- component guard `0/0`;
- threshold grid `0.50,0.60,0.70,0.75,0.80,0.85,0.90`;
- F2 inference filters retained:
  rank `<=2`, gradient ratio `>=2.0`, distance ratio `>=0.12`,
  slenderness `>=4.0`, fill `<=0.80`.

F3 promotion criteria:

- Accepted rate must exceed R212-F2's `1/32` without returning to R212-F1's
  bad accepts.
- `13867.png` and `15040.png` must remain unaccepted unless a different
  accepted candidate is clearly positive by R201 metrics.
- Useful cases such as `15067.png` should be recovered if the model can do so
  without hurting mean Boundary IoU/F1 and Surface Dice 2px.
- Mean gap FP reduction and boundary/surface gains must exceed R212-F2 before
  any full-val consideration.

## R212-F3 Interim Monitor

Remote screen `r212_f3_shape_features_val32` remains alive and error-free.
The run has completed train64 candidate generation and entered val32 candidate
generation. No final summary has been produced yet.

Current evidence:

- train candidate phase completed in about `16:45` according to the remote log;
- val candidate generation has started and reached the early val cases;
- no traceback or runtime error is present in
  `outputs/bridge_logs/r212_f3_shape_features_val32.log`;
- `clean-test-v2` has not been used.

Next required action:

- wait for `outputs/analysis/r212_f3_remote_val32_val_summary.json`;
- sync summary/per-image/candidates/checkpoint/log;
- compare F3 against F2 and F1, especially accepted cases
  `13867.png`, `15040.png`, `15067.png`, and `15446.png`.

## R212-F3 Val32 Result

Remote R212-F3 completed on original train/val only. `clean-test-v2` was not
used.

Artifacts synced locally:

- `outputs/analysis/r212_f3_remote_val32_train_candidates.csv`
- `outputs/analysis/r212_f3_remote_val32_val_candidates.csv`
- `outputs/analysis/r212_f3_remote_val32_val_summary.json`
- `outputs/analysis/r212_f3_remote_val32_val_per_image.csv`
- `outputs/bridge_logs/r212_f3_shape_features_val32.log`
- `outputs/r211_f1_learned_acceptance_gate/r212_f3_shape_features_hgb_remote_val32.joblib`

Candidate labels:

- train candidates: `304`
- train positives: `96`
- train positive rate: `0.315789`
- val candidates: `137`
- val positives: `39`
- val positive rate: `0.284672`

Best threshold: `0.80`.

Mean deltas vs R110 anchor on val32:

- accepted image rate: `0.031250`
- Dice: `-0.0000003`
- IoU: `-0.0000005`
- Recall: `-0.0000094`
- Boundary IoU: `+0.0000074`
- Boundary F1: `+0.0000100`
- Surface Dice 2px: `+0.0000194`
- Surface Dice 5px: `-0.0000070`
- HD95 px: `0.000000`
- ASSD px: `-0.0000174`
- gap-region FP rate: `-0.0000220`
- component merge rate: `0.000000`
- component count MAE: `0.000000`
- predicted component count delta: `0.000000`
- instance merge count delta: `0.000000`

Gate result: `true`.

Accepted case:

- `15446.png`: accepted candidate rank `1`, probability `0.817625`,
  Boundary IoU `+0.000238`, Boundary F1 `+0.000321`,
  Surface Dice 2px `+0.000621`, gap FP `-0.000705`,
  Dice `-0.000009`, Recall `-0.000300`, component count MAE `0.0`.

Known R212-F1/F2 audit cases:

- `13867.png`: not accepted.
- `15040.png`: not accepted; the new shape/contact features suppress the
  unsafe high-score accept seen in earlier variants.
- `15067.png`: still missed despite positive candidates. The strongest useful
  candidates include rank `2` with probability `0.583503` and rank `3` with
  probability `0.661703`, both below the selected `0.80` threshold.

## R212-F3 Interpretation

Verdict: `technical_pass_not_promoted`.

R212-F3 achieved the narrow risk-control goal but did not improve over R212-F2
in a meaningful way. It again accepted only `1/32` validation images, with
near-identical mean metric deltas to F2. The added GT-free shape/contact
features help suppress unsafe candidates such as `15040.png`, but the selected
threshold and F2-style filters remain too conservative to recover useful cases
such as `15067.png`.

Decision:

- Do not apply R212-F3 to `clean-test-v2`.
- Do not promote R212-F3 to full original-val.
- Do not continue by only adding more features or tightening thresholds.
- Continue to R212-F4 only after adding a better threshold/case audit and
  selection objective.

Recommended R212-F4 direction:

1. Save per-threshold accepted-image lists and accepted-candidate summaries,
   not only the best-threshold per-image CSV.
2. Inspect thresholds `0.50`, `0.75`, and `0.80` to identify exactly which
   cases make the gate fail or pass.
3. Modify selection so a passed threshold must have a meaningful accepted rate
   and stronger gap/boundary improvement, while still rejecting obvious unsafe
   edits.
4. Consider relaxing `rank <= 2` only with stronger feature/probability guards,
   because useful candidates exist at higher rank in some cases.
5. Keep all tuning on original train/val only; `clean-test-v2` remains locked.

## R212-F4 Audit Instrumentation Prep

Implemented the first R212-F4 prerequisite in
`scripts/run_r211_f1_learned_acceptance_gate.py`:

- added default-off `--threshold-audit-csv`;
- when enabled, the script writes one row per accepted image per threshold;
- each row includes threshold, gate status, accepted candidate summary, and
  core R201 deltas for overlap, boundary, surface, gap, component, and instance
  merge metrics.

Validation:

- `python -m py_compile scripts/run_r211_f1_learned_acceptance_gate.py` passed.

This is instrumentation only. No new model result, full-val run, or
`clean-test-v2` evaluation has been produced from F4 yet.

## R212-F4 Threshold Audit Result

Remote R212-F4 completed on original train/val only. `clean-test-v2` was not
used. The run reused the R212-F3 settings and enabled only the new
`--threshold-audit-csv` output, so the main metrics match F3.

Artifacts synced locally:

- `outputs/analysis/r212_f4_remote_val32_train_candidates.csv`
- `outputs/analysis/r212_f4_remote_val32_val_candidates.csv`
- `outputs/analysis/r212_f4_remote_val32_val_summary.json`
- `outputs/analysis/r212_f4_remote_val32_val_per_image.csv`
- `outputs/analysis/r212_f4_remote_val32_threshold_audit.csv`
- `outputs/bridge_logs/r212_f4_threshold_audit_val32.log`
- `outputs/r211_f1_learned_acceptance_gate/r212_f4_threshold_audit_hgb_remote_val32.joblib`

Best threshold: `0.80`.

Mean deltas vs R110 anchor at the selected threshold:

- accepted image rate: `0.031250`
- Dice: `-0.0000003`
- Boundary IoU: `+0.0000074`
- Boundary F1: `+0.0000100`
- Surface Dice 2px: `+0.0000194`
- gap-region FP rate: `-0.0000220`
- ASSD px: `-0.0000174`

Threshold audit:

- `0.50`: accepted `13480.png`, `15067.png`, and `15446.png`; gate failed.
- `0.60`/`0.70`: accepted `13480.png` and `15446.png`; gate failed.
- `0.75`: accepted `13480.png` and `15446.png`; gate passed by mean values
  but still includes a bad edit.
- `0.80`: accepted only `15446.png`; gate passed but effect size stayed tiny.

Key case diagnosis:

- `15067.png` is a useful missed case at low threshold:
  Boundary IoU `+0.001604`, Boundary F1 `+0.001928`,
  Surface Dice 2px `+0.001504`, gap FP `-0.001545`,
  ASSD `-0.007158`.
- `15446.png` is a safe small edit:
  Boundary IoU `+0.000238`, Surface Dice 2px `+0.000621`,
  gap FP `-0.000705`, ASSD `-0.000558`.
- `13480.png` is the harmful accept:
  at threshold `0.75`, Dice `-0.000589`, Boundary IoU `-0.000127`,
  Surface Dice 2px `-0.000606`, ASSD `+0.011560`, and no gap-FP gain.

GT-free separation signal:

- harmful `13480.png` accepted candidate has `cut_frac_component=0.010841`;
- useful `15067.png` accepted candidate has `cut_frac_component=0.001751`;
- useful `15446.png` accepted candidate has `cut_frac_component=0.004518`.

Decision:

- R212-F4 is an audit result, not a promoted model.
- Do not apply to `clean-test-v2`.
- Test R212-F5 with a stricter component-fraction filter,
  `--max-cut-frac-component 0.006`, to reject `13480.png` while preserving
  `15067.png` and `15446.png`.

## R212-F5 Cut-Fraction Filter Launch

Remote R212-F5 launched on original train/val only. `clean-test-v2` is not
used.

Purpose:

- verify whether `--max-cut-frac-component 0.006` can remove the harmful
  `13480.png` accept while retaining useful smaller neck cuts such as
  `15067.png` and `15446.png`.

Remote run:

- screen: `r212_f5_cutfrac006_val32`
- log: `outputs/bridge_logs/r212_f5_cutfrac006_val32.log`
- summary target: `outputs/analysis/r212_f5_remote_val32_val_summary.json`
- threshold audit target:
  `outputs/analysis/r212_f5_remote_val32_threshold_audit.csv`
- checkpoint:
  `outputs/r211_f1_learned_acceptance_gate/r212_f5_cutfrac006_hgb_remote_val32.joblib`

Key changes vs F4:

- `--max-cut-frac-component 0.006`
- threshold grid `0.50,0.55,0.60,0.65,0.70,0.75,0.80`

Promotion requirement:

- recover `15067.png` or at least improve accepted rate beyond F3/F4;
- keep `13480.png`, `13867.png`, and `15040.png` out of accepted cases;
- improve boundary/surface/gap mean deltas more than F3/F4;
- avoid recall/Dice collapse and keep component metrics unchanged or better.

## R212-F5 Cut-Fraction Filter Val32 Result

Remote R212-F5 completed on original train/val only. `clean-test-v2` was not
used.

Artifacts synced locally:

- `outputs/analysis/r212_f5_remote_val32_train_candidates.csv`
- `outputs/analysis/r212_f5_remote_val32_val_candidates.csv`
- `outputs/analysis/r212_f5_remote_val32_val_summary.json`
- `outputs/analysis/r212_f5_remote_val32_val_per_image.csv`
- `outputs/analysis/r212_f5_remote_val32_threshold_audit.csv`
- `outputs/bridge_logs/r212_f5_cutfrac006_val32.log`
- `outputs/r211_f1_learned_acceptance_gate/r212_f5_cutfrac006_hgb_remote_val32.joblib`

Best threshold: `0.50`.

Mean deltas vs R110 anchor on val32:

- accepted image rate: `0.062500`
- Dice: `+0.0000078`
- IoU: `+0.0000139`
- Recall: `-0.0000119`
- Boundary IoU: `+0.0000576`
- Boundary F1: `+0.0000703`
- Surface Dice 2px: `+0.0000664`
- Surface Dice 5px: `+0.0000007`
- HD95 px: `0.000000`
- ASSD px: `-0.0002411`
- gap-region FP rate: `-0.0000703`
- component merge rate: `0.000000`
- component count MAE: `0.000000`
- predicted component count delta: `0.000000`
- instance merge count delta: `0.000000`

Accepted cases at the selected threshold:

- `15067.png`: Boundary IoU `+0.001604`, Boundary F1 `+0.001928`,
  Surface Dice 2px `+0.001504`, Surface Dice 5px `+0.000246`,
  gap FP `-0.001545`, ASSD `-0.007158`, Dice `+0.000260`,
  component count MAE `0.0`.
- `15446.png`: Boundary IoU `+0.000238`, Boundary F1 `+0.000321`,
  Surface Dice 2px `+0.000621`, gap FP `-0.000705`,
  ASSD `-0.000558`, Dice `-0.000009`, component count MAE `0.0`.

Risk-case audit:

- `13480.png`: no longer accepted after `max_cut_frac_component=0.006`.
- `13867.png`: not accepted.
- `15040.png`: not accepted.

Interpretation:

R212-F5 is the first R212 val32 smoke that satisfies the intended local
direction: accepted rate improves beyond F3/F4, the known harmful accept is
removed, and overlap/boundary/surface/gap/ASSD mean deltas are all favorable
with component metrics unchanged. The gains are still small because only
`2/32` images are edited, but the acceptance policy is now cleaner than F1-F4.

Verdict: `val32_directional_pass_not_clean_test_ready`.

Decision:

- Do not apply R212-F5 to `clean-test-v2`.
- Promote only to original full-val verification, not final test.
- Full-val requirement:
  - accepted rate must remain nonzero and meaningful;
  - no reappearance of obvious bad accepts like `13480.png`, `13867.png`, or
    `15040.png` analogues;
  - Boundary IoU/F1, Surface Dice 2px/5px, ASSD, and gap FP should remain
    better than the R110 anchor;
  - Dice/IoU/Recall drop must stay negligible;
  - component merge/count metrics must not worsen.

## R212-F5 Full-Val Candidate-Level Precheck

While the official original-val eval-only run was still computing full R201
mask metrics, a read-only candidate-level precheck was run on the server using
the same val32 F5 checkpoint and the materialized full-val candidate CSV. This
does not replace full-val evaluation because it does not generate final masks
or recompute mean R201 metrics, but it is useful for identifying likely bad
accepts before the official threshold audit is written.

Artifacts:

- `scripts/audit_r212_candidate_precheck.py`
- `outputs/analysis/r212_f5_evalonly_fullval_candidate_precheck.json`
- `outputs/analysis/r212_f5_evalonly_fullval_candidate_precheck.csv`

Precheck summary:

| Threshold | Accepted images | Label-negative accepted | Risk accepted |
| --- | ---: | ---: | ---: |
| 0.50 | 10 | 7 | 9 |
| 0.55 | 10 | 7 | 9 |
| 0.60 | 8 | 7 | 7 |
| 0.65 | 8 | 7 | 7 |
| 0.70 | 7 | 6 | 6 |
| 0.75 | 3 | 2 | 2 |
| 0.80 | 1 | 0 | 1 |

Likely risk images at lower thresholds:

- `1620.png`
- `3058.png`
- `3468.png`
- `6605.png`
- `7259.png`
- `8253.png`

Interpretation:

The val32 F5 checkpoint does not appear to generalize cleanly across full
original val at low thresholds: the candidate-level precheck predicts that
many accepted images would be label-negative or have negative Dice/boundary
deltas. This means the official full-val result is likely to choose a high
threshold with tiny effect, or fail the boundary gate at lower thresholds.

Decision:

- Continue waiting for the official eval-only full-val summary; do not decide
  from this precheck alone.
- When the official threshold audit is available, specifically check whether
  `6605.png`, `1620.png`, `3468.png`, and `8253.png` are accepted.
- If these risk cases are accepted, do not promote F5; move to an R213
  acceptance-gate repair or candidate replay/audit tool.

## R212-F5 Eval-Only Full Original-Val Result

The official eval-only full-val run completed on original `TSRS_RSNA-Epiphysis`
validation only. It loaded the val32 F5 checkpoint and did not use
`clean-test-v2`.

Artifacts:

- `outputs/analysis/r212_f5_evalonly_fullval_val_summary.json`
- `outputs/analysis/r212_f5_evalonly_fullval_threshold_audit.csv`
- `outputs/analysis/r212_f5_evalonly_fullval_val_per_image.csv`
- `outputs/analysis/r212_f5_evalonly_fullval_val_candidates.csv`
- `outputs/bridge_logs/r212_f5_evalonly_fullval.log`

Best threshold: `0.80`.

Gate status: `pass`.

Mean deltas vs full original-val R110 train/val anchor:

- accepted image rate: `0.010417` (`1/96`, only `15446.png`)
- Dice: `-0.00000010`
- IoU: `-0.00000016`
- Recall: `-0.00000312`
- Boundary IoU: `+0.00000248`
- Boundary F1: `+0.00000334`
- Surface Dice 2px: `+0.00000647`
- Surface Dice 5px: `-0.00000233`
- HD95 px: `0.00000000`
- ASSD px: `-0.00000582`
- gap-region FP rate: `-0.00000734`
- component merge rate: `0.00000000`
- component count MAE: `0.00000000`
- instance merge count: `0.00000000`
- predicted component count: `0.00000000`

Threshold audit:

| Threshold | Accepted images | Gate pass | Risk image accepts |
| --- | ---: | ---: | --- |
| 0.50 | 9 | 0 | `1620`, `3058`, `3468`, `6605`, `7259`, `8253` |
| 0.55 | 9 | 0 | `1620`, `3058`, `3468`, `6605`, `7259`, `8253` |
| 0.60 | 7 | 0 | `1620`, `3058`, `3468`, `6605`, `7259`, `8253` |
| 0.65 | 7 | 0 | `1620`, `3058`, `3468`, `6605`, `7259`, `8253` |
| 0.70 | 6 | 0 | `1620`, `3058`, `3468`, `6605`, `8253` |
| 0.75 | 2 | 0 | `6605` |
| 0.80 | 1 | 1 | none |

Interpretation:

R212-F5 generalizes safely but weakly. The selected threshold removes the
precheck risk cases and preserves component metrics, but the accepted rate is
only `1/96` and the mean improvements are on the order of `1e-6` to `1e-5`.
This is effectively a near no-op, not a meaningful project-level improvement
on bone-gap adhesion or boundary quality.

Decision:

- Do not promote the val32 F5 checkpoint to `clean-test-v2`.
- Do not treat eval-only full-val as evidence of a new best model.
- Continue waiting for the strict full-train/full-val F5 run, because it may
  learn a different acceptance boundary from all train candidates.
- If strict full-val also selects a high-threshold near no-op, close R212-F5
  and move to R213 with either stronger candidate generation or an explicit
  nonzero-edit validation objective plus risk-case rejection.

## R213 Validation Gate Audit Tool

To prevent near no-op validation passes from being mistaken for real project
progress, a report-only promotion gate was added.

Artifacts:

- `scripts/audit_r213_validation_gate.py`
- `outputs/analysis/r213_gate_audit_r212_f5_evalonly_fullval.json`
- `research-workflow/refine-logs/R213_GATE_AUDIT_R212_F5_EVALONLY_FULLVAL.md`

The audit checks:

- summary gate pass;
- accepted image rate is meaningful, default `>=0.03`;
- Dice/IoU/Recall drops stay below tolerance;
- Boundary IoU, Boundary F1, and Surface Dice 2px improve by at least `1e-4`;
- known risk images are not accepted;
- optional gap FP improvement and component non-worsening.

Applied to R212-F5 eval-only full-val, the decision is `do_not_promote`.

Failed checks:

- `accepted_rate_meaningful`: accepted only `1/96`;
- `boundary_iou_meaningful`: `+0.00000248`, below `1e-4`;
- `boundary_f1_meaningful`: `+0.00000334`, below `1e-4`;
- `surface2_meaningful`: `+0.00000647`, below `1e-4`.

Passed checks:

- no known risk image was accepted at the selected threshold;
- Dice, IoU, and Recall were non-worse;
- gap-region FP improved slightly;
- component merge/count and predicted component count were non-worse.

Decision:

- Use this gate for the strict full-train/full-val F5 result when it finishes.
- R213 should be designed to pass this stricter validation gate before any
  visual audit or clean-test-v2 consideration.

## R213 Candidate-Signal Audit on R212-F5 Full-Val Candidates

Because the strict full-train/full-val F5 run is still computing train
candidates, a candidate-level diagnostic was run on the already materialized
eval-only full-val candidate CSV. This is validation-only analysis and does
not replace mask-derived R201 evaluation.

Artifact:

- `outputs/analysis/r213_candidate_signal_audit_r212_f5_evalonly_fullval.json`

Summary:

| Subset | Rows | Images | Median prob | Max prob |
| --- | ---: | ---: | ---: | ---: |
| all candidates | 433 | 92 | 0.149575 | 0.894881 |
| label-positive candidates | 97 | 52 | 0.261876 | 0.849673 |
| strict useful candidates | 75 | 41 | 0.234800 | 0.849673 |
| risk candidates | 406 | 92 | 0.147789 | 0.894881 |
| F5 static-filter pass | 34 | 27 | 0.166822 | 0.817625 |
| useful + static-filter pass | 6 | 6 | 0.565814 | 0.817625 |
| risk + static-filter pass | 33 | 26 | 0.134346 | 0.817625 |

Important observations:

- Full original val contains useful candidates: `75` strict useful candidates
  across `41` images.
- F5's current static filters are too restrictive for recall: only `6` useful
  candidates survive the filter stack.
- The learned acceptor is also conservative: `63/75` strict useful candidates
  have probability below `0.50`, and `71/75` are below `0.80`.
- Lower thresholds recover some useful edits but also accept many risk cases.
  At threshold `0.50`, candidate precheck accepts `10` images but only `3`
  are strict useful/positive while `9` are risk.

Main missed-useful reasons:

- `prob<0.50`: `63`
- `prob<0.80`: `71`
- `rank>2`: `43`
- `slenderness<4`: `35`
- `fill>0.80`: `25`
- `cut_frac_component>0.006`: `23`
- `gradient<2`: `19`
- `dist<0.12`: `11`

Interpretation:

R212-F5 is not failing because the validation set lacks repairable bone-gap
or boundary candidates. It is failing because the current F5 acceptance policy
has low useful-candidate recall, while simple threshold relaxation admits
risk candidates. Therefore, the next useful step is not another higher
threshold or a blind stricter filter. R213 should explicitly separate two
subproblems:

1. recover useful-candidate recall by relaxing or learning over the rank,
   slenderness, fill, and cut-fraction filters;
2. add a risk-rejection layer that targets the known false accepts
   (`1620`, `3058`, `3468`, `6605`, `7259`, `8253`, `15040`, etc.).

R213 design implication:

- Keep the R213 validation-gate audit as the promotion criterion.
- Add a candidate-recall stage or two-stage classifier rather than only tuning
  the final probability threshold.
- Require nonzero meaningful full-val effect before any visual audit or
  clean-test-v2 consideration.

## R213 Candidate-Policy Search

A candidate-policy search tool was added to quantify whether simple threshold
and static-filter tuning can recover useful candidates while controlling risk.

Artifacts:

- `scripts/search_r213_candidate_policy.py`
- `outputs/analysis/r213_candidate_policy_search_r212_f5_evalonly_fullval.json`
- `outputs/analysis/r213_candidate_policy_search_r212_f5_evalonly_fullval.csv`

Search space:

- probability threshold: `0.20` to `0.80`;
- `max_cut_frac_component`: `0.006` to `0.020`;
- `max_candidate_rank`: `2` to `5`;
- `min_gradient_ratio`: `1.4` to `2.0`;
- `min_dist_ratio`: `0.09` to `0.12`;
- `min_slenderness`: `1.0` to `4.0`;
- `max_fill`: `0.80` to `1.00`.

Risk definition was adjusted to hard risk only:

- Dice `< -5e-4`;
- IoU `< -8e-4`;
- Recall `< -8e-4`;
- Boundary IoU / Boundary F1 / Surface Dice 2px below `0`.

Key frontier results:

| Constraint | Best strict useful | Accepted | Risk | Precision useful |
| --- | ---: | ---: | ---: | ---: |
| risk <= 0 | 1 | 1 | 0 | 1.000 |
| risk <= 1 | 2 | 3 | 1 | 0.667 |
| risk <= 2 | 2 | 3 | 1 | 0.667 |
| risk <= 10 | 8 | 20 | 10 | 0.400 |
| risk <= 20 | 14 | 33 | 17 | 0.424 |
| useful >= 10 | 10-12 | 24-28 | 13-14 | 0.400-0.429 |
| useful >= 15 | 15 | 41-43 | 24 | 0.349-0.366 |

Interpretation:

Simple threshold/static-filter search cannot solve the R212-F5 bottleneck.
The current F5 probability model and morphology filters have a poor frontier:
safe settings are near no-op, while useful-recall settings admit too many
risk edits. This validates the earlier candidate-signal audit and makes the
next direction sharper.

Decision:

- Do not spend more runs on plain threshold/filter sweeps around F5.
- R213 should introduce a new rejection signal, not just tune F5:
  - a two-stage useful-vs-risk classifier trained on full train candidates;
  - or a hard-risk veto using validation-derived failure features;
  - or a different candidate generation/edit family with cleaner candidates.
- Any R213 promotion must pass `scripts/audit_r213_validation_gate.py` on
  original full val before visual audit or clean-test-v2 use.

## R213 Two-Stage Gate Preparation

R213 was formalized as a two-stage candidate gate:

- Stage A: useful-recall model;
- Stage B: hard-risk rejection model.

Artifacts:

- `research-workflow/refine-logs/R213_TWO_STAGE_RISK_REJECTION_PLAN.md`
- `scripts/train_r213_two_stage_candidate_gate.py`

The script:

- trains useful and hard-risk classifiers from candidate CSV rows;
- evaluates useful/risk threshold grids on validation candidate rows;
- reports risk-constrained frontiers;
- writes JSON/CSV only;
- does not write masks and does not use `clean-test-v2`.

Verification:

- local `py_compile` passed;
- local selfcheck on eval-only full-val candidates passed, then temporary
  selfcheck outputs were deleted to avoid confusing them with experiment
  evidence;
- script was synced to the server and remote `py_compile` passed.

Blocked input:

- strict R212-F5 full-train/full-val is still running, so the full train
  candidate CSV needed for the real R213 candidate-level training is not yet
  available.

Next action after strict F5 finishes:

1. sync `r212_f5_remote_fullval_train_candidates.csv` and val candidates;
2. run `scripts/train_r213_two_stage_candidate_gate.py` using train candidates
   for model fitting and original val candidates for candidate-level
   selection;
3. only if the candidate-level frontier improves, build a mask-level R213
   evaluator and run original full-val R201 metrics.

## R213 Two-Stage Selfprobe and Grouped-CV Diagnostic

While strict R212-F5 full-train/full-val was still generating train candidates,
the two-stage candidate gate was tested on the already available eval-only
full-val candidate CSV.

Artifacts:

- `outputs/analysis/r213_two_stage_candidate_gate_val_selfprobe.json`
- `outputs/analysis/r213_two_stage_candidate_gate_val_selfprobe.csv`
- `outputs/analysis/r213_two_stage_candidate_gate_val_groupedcv5.json`
- `outputs/analysis/r213_two_stage_candidate_gate_val_groupedcv5.csv`

Important caution:

- `val_selfprobe` uses the same candidate CSV for training and validation.
  It is a pipeline/self-fit upper-bound probe only and is not evidence of
  generalization.
- `val_groupedcv5` uses five-fold image-grouped CV, so candidates from the
  same image do not appear in both training and validation folds. This is the
  more relevant diagnostic.

Selfprobe result:

- best settings accept `41-42` images;
- strict useful `41`;
- hard risk `0`;
- precision useful nearly `1.0`.

Grouped-CV result:

| Constraint | Best strict useful | Accepted | Hard risk | Precision useful |
| --- | ---: | ---: | ---: | ---: |
| risk <= 0 | 0 | 1-2 | 0 | 0.000 |
| risk <= 1 | 1 | 3-4 | 1 | 0.250-0.333 |
| risk <= 10 | 4 | 16-20 | 9-10 | 0.200-0.250 |
| useful >= 5 | 5-6 | 20-27 | 11-12 | 0.208-0.250 |
| useful >= 10 | 10 | 48-62 | 26-38 | 0.161-0.208 |

Interpretation:

The two-stage classifier can memorize the candidate table, but it does not
generalize across validation images with the current feature set. This means
R213 cannot be promoted directly to mask-level evaluation from the selfprobe.
The current candidate features are still insufficient to separate useful
edits from hard-risk edits under image-grouped validation.

Decision:

- Keep `scripts/train_r213_two_stage_candidate_gate.py` as infrastructure.
- Do not launch mask-level R213 from the selfprobe result.
- When strict F5 train candidates finish, run the real train->val candidate
  experiment. If it resembles grouped-CV rather than selfprobe, close the
  current feature-only two-stage gate and move to cleaner candidate generation
  or image-context features.

Protocol metadata update:

- `scripts/train_r213_two_stage_candidate_gate.py` now writes
  `evidence_level`, `protocol_warning`, `clean_test_v2_used=false`, and
  `writes_masks=false` into every JSON report.
- `r213_two_stage_candidate_gate_val_selfprobe.json` is explicitly marked as
  `candidate_selffit_probe`.
- `r213_two_stage_candidate_gate_val_groupedcv5.json` is explicitly marked as
  `candidate_grouped_cv_diagnostic`.

This prevents self-fit probes or candidate-only diagnostics from being
mistaken for mask-level R201 validation evidence.
