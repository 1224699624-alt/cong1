# R216 Soft Seam-Action Progress

Date: 2026-07-07

## Purpose

R215 confirmed that background-channel connectivity is a useful diagnostic
signal for bone-gap separation, but hard deletion candidates do not generalize
safely enough. R216 changes the action space rather than sweeping thresholds:

- keep R110 as the anchor;
- regenerate the same train/val-only seam/neck candidates;
- test soft seam actions by removing only the most seam-like fraction of each
  candidate cut (`0.25`, `0.50`, `0.75`, `1.00`);
- measure whether partial actions reduce over-erosion while retaining
  boundary/gap gains.

This is candidate-action diagnostic work only. It writes no masks and does not
use `clean-test-v2`.

## Implemented

Added:

- `scripts/audit_r216_soft_seam_action_candidates.py`

The script outputs:

- candidate-action CSV;
- summary JSON;
- no final masks.

The action score favors pixels that are:

- close to the current anchor boundary / seam side;
- supported by local image gradient;
- adjacent to local background channels.

It computes the same quick R201-style metrics used by R215 diagnostics:

- Dice / IoU / Recall;
- Boundary IoU / Boundary F1;
- gap-region FP rate;
- component count MAE / delta.

It also records train/val-only GT diagnostics such as cut GT foreground/gap
fraction. These diagnostics are not deployable rules.

## Local Smoke

Command:

```bash
python scripts/audit_r216_soft_seam_action_candidates.py \
  --split val \
  --limit 8 \
  --source-candidate-csv outputs/analysis/r212_f5_evalonly_fullval_val_candidates.csv \
  --max-candidates-generated 8 \
  --max-components-per-image 3 \
  --action-fracs 0.25,0.50,0.75,1.00 \
  --output-csv outputs/analysis/r216_soft_seam_action_val_smoke_candidates.csv \
  --output-json outputs/analysis/r216_soft_seam_action_val_smoke_summary.json
```

Smoke result:

- rows: `12`
- images with candidates: `1`
- quick-useful: `11`
- hard-risk: `1`

By action fraction:

- `0.25`: `3/3` useful, `0` risk, Boundary IoU `+0.001236`, gap FP `-0.000869`
- `0.50`: `3/3` useful, `0` risk, Boundary IoU `+0.001309`, gap FP `-0.001071`
- `0.75`: `3/3` useful, `0` risk, Boundary IoU `+0.002140`, gap FP `-0.001998`
- `1.00`: `2/3` useful, `1` risk, Boundary IoU `+0.002238`, gap FP `-0.002780`

Early interpretation: partial actions appear to reduce hard-risk relative to
full hard deletion while preserving the intended gap/boundary gains. This is
only a tiny smoke result and must be verified on full original val.

## Remote Full-Val Run

Launched:

- screen: `r216_soft_seam_fullval`
- log: `outputs/bridge_logs/r216_soft_seam_action_fullval.log`
- candidates: `outputs/analysis/r216_soft_seam_action_fullval_candidates.csv`
- summary: `outputs/analysis/r216_soft_seam_action_fullval_summary.json`

Command uses original `val` only and `--flush-every 5`.

Completed:

- candidate-action rows: `1060`
- images: `92`
- candidate groups with all action fractions: `265`

Overall by action fraction:

- `0.25`: useful `114/265`, hard-risk `143/265`, mean Dice `-0.000018`, Boundary IoU `+0.000090`, gap FP `-0.000247`
- `0.50`: useful `102/265`, hard-risk `141/265`, mean Dice `-0.000037`, Boundary IoU `+0.000056`, gap FP `-0.000416`
- `0.75`: useful `93/265`, hard-risk `158/265`, mean Dice `-0.000072`, Boundary IoU `+0.000073`, gap FP `-0.000541`
- `1.00`: useful `73/265`, hard-risk `177/265`, mean Dice `-0.000166`, Boundary IoU `-0.000123`, gap FP `-0.000846`

Oracle best partial action per candidate group:

- useful `127/265`
- hard-risk `120/265`
- mean Dice `+0.000015`
- Boundary IoU `+0.000344`
- Boundary F1 `+0.000469`
- gap FP `-0.000484`

Full hard deletion on the same candidate groups:

- useful `73/265`
- hard-risk `177/265`
- mean Dice `-0.000166`
- Boundary IoU `-0.000123`
- Boundary F1 `-0.000154`
- gap FP `-0.000846`

Safe-channel diagnostic group:

- rows: `345`
- quick-useful: `299`
- hard-risk: `27`
- mean Dice `+0.000130`
- Boundary IoU `+0.000881`
- Boundary F1 `+0.001169`
- gap FP `-0.000955`
- mean cut GT foreground fraction `0.163851`
- mean cut GT gap fraction `0.817562`

Interpretation: R216 validates the action-space change. Partial seam actions
are substantially safer than full deletion and keep useful boundary/gap signal.
This is still not deployable evidence, because the best subsets use train/val
labels and GT diagnostics. The next step is an action-level grouped-CV gate.

## Promotion Gate

R216 can only move forward if full original-val evidence shows:

- partial action fractions lower hard-risk rate versus `1.00` hard deletion;
- useful/risk frontier improves over R215 and R213 fused gates;
- Dice / IoU / Recall are not materially degraded;
- boundary/gap gains remain nontrivial;
- component count MAE does not worsen.

Do not apply R216 to `clean-test-v2` unless a later train/val gate passes and a
mask-level R201 validation step is implemented.

## Action-Level Grouped-CV Gate

Implemented:

- `scripts/train_r216_soft_seam_action_gate.py`

Artifacts:

- `outputs/analysis/r216_soft_seam_action_gate_groupedcv5_hgb_quick.json`
- `outputs/analysis/r216_soft_seam_action_gate_groupedcv5_logreg_quick.json`
- `outputs/analysis/r216_soft_seam_action_gate_groupedcv5_hgb_safe.json`
- `outputs/analysis/r216_soft_seam_action_gate_groupedcv5_logreg_safe.json`

Results:

- HGB quick/hard: accepts `91`, useful `41`, risk `48`, Boundary IoU `-0.000046`
- LogReg quick/hard: accepts `7`, useful `4`, risk `2`, Boundary IoU `+0.000330`
- HGB safe/overerosion: accepts `82`, useful `27`, risk `53`, Boundary IoU `-0.000128`
- LogReg safe/overerosion: accepts `2`, useful `1`, risk `1`, Boundary IoU `+0.000249`

Decision:

- R216 validates the action-space idea, but the current inference features do
  not yet support a safe deployable gate.
- Do not write R216 masks and do not run clean-test-v2.
- The next version should not be another threshold sweep. It should add
  stronger GT-free over-erosion / true-seam discrimination, likely through
  local image-context or patch-level features.

Recommended R217:

1. Keep R216 partial seam actions.
2. Add local patch/image context around the candidate action:
   - intensity contrast across both sides of the seam;
   - distance-to-anchor-boundary and medial-axis statistics;
   - local background-channel width/continuity;
   - whether the cut separates two plausible bone instances or enters one bone body.
3. Train/evaluate grouped-CV action gate again.
4. Promote only if low-risk frontier accepts enough useful actions with
   positive boundary/gap deltas and non-worse Dice/Recall.
