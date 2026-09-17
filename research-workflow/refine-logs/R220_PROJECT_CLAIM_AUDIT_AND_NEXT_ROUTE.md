# R220 Project Claim Audit and Next Route

Date: 2026-07-08

## Purpose

Audit the current YOLO+SAM epiphysis segmentation improvement project after
R216-R219. The goal is not paper writing. The goal is to decide, from evidence,
whether the current branch supports the project claim and what experiment route
should be pursued next.

Project claim target:

- keep Dice/IoU competitive with R110/ARAA;
- improve bone-gap adhesion, adjacent-bone false connections, boundary quality,
  and anatomical consistency;
- avoid apparent gains caused by over-erosion or missed bone foreground.

## Current R201 Baselines

All values below are from:

- `outputs/analysis/r201_unified_eval/r201_unified_comparison_table.csv`

### ARAA DANet epoch98

- Dice: `0.911766`
- IoU: `0.838436`
- Boundary IoU: `0.235265`
- Boundary F1: `0.374350`
- Surface Dice 2px: `0.677752`
- Surface Dice 5px: `0.917918`
- HD95: `6.596897`
- ASSD: `2.343013`
- gap-region FP rate: `0.201259`
- component merge rate: `0.679012`
- component count MAE: `2.086420`

### R110 Current Best

- Dice: `0.917723`
- IoU: `0.848570`
- Boundary IoU: `0.251896`
- Boundary F1: `0.394721`
- Surface Dice 2px: `0.684156`
- Surface Dice 5px: `0.915014`
- HD95: `7.220072`
- ASSD: `2.419143`
- gap-region FP rate: `0.201358`
- component merge rate: `0.790123`
- component count MAE: `2.506173`

R110 improves overlap and boundary metrics over ARAA, but worsens component
merge and component count error.

### R202 nnU-Net Risk Audit

- Dice: `0.923573`
- IoU: `0.858508`
- Boundary IoU: `0.270431`
- Boundary F1: `0.417773`
- Surface Dice 2px: `0.713582`
- Surface Dice 5px: `0.930538`
- HD95: `6.113146`
- ASSD: `2.083621`
- gap-region FP rate: `0.218767`
- component merge rate: `0.679012`
- component count MAE: `1.827160`

R202 is not the main target to beat on Dice/IoU, but it is a risk audit showing
that stronger generic segmentation can outperform R110 on many metrics while
worsening gap FP.

## R216-R218 Evidence

### R216 Soft Seam Action

Artifact:

- `outputs/analysis/r216_soft_seam_action_fullval_summary.json`

Full deletion (`action_frac=1.00`) is unsafe:

- useful `73/265`
- hard-risk `177/265`
- mean Boundary IoU `-0.000123`

Partial actions are safer:

- `0.25`: useful `114/265`, hard-risk `143/265`, Boundary IoU `+0.000090`
- `0.50`: useful `102/265`, hard-risk `141/265`, Boundary IoU `+0.000056`
- `0.75`: useful `93/265`, hard-risk `158/265`, Boundary IoU `+0.000073`

GT diagnostic safe-channel group is strong:

- rows: `345`
- useful: `299`
- hard-risk: `27`
- Dice delta: `+0.000130`
- Boundary IoU delta: `+0.000881`
- Boundary F1 delta: `+0.001169`
- gap FP delta: `-0.000955`

Interpretation:

- There is real signal: background/seam-oriented partial actions can reduce
  bridge/gap errors.
- But the strongest safe-channel signal uses GT diagnostics and is not a
  deployable rule.

### R216-R218 Gates

Best deployable-ish low-risk reference remains R216 LogReg quick/hard:

- risk<=2: accepted `7`, useful `4`, risk `2`, Boundary IoU `+0.000330`
- risk<=5: accepted `10`, useful `6`, risk `3`, Boundary IoU `+0.000229`
- risk<=10: accepted `18`, useful `10`, risk `7`, Boundary IoU `+0.000380`

R217 hand-crafted patch context did not improve:

- risk<=5: accepted `10`, useful `4`, risk `4`, Boundary IoU `+0.000110`
- risk<=10: accepted `18`, useful `7`, risk `9`, Boundary IoU `+0.000065`

R218 local patch CNN did not improve:

- risk<=5: accepted `7`, useful `2`, risk `5`, Boundary IoU `-0.000042`
- risk<=10: accepted `13`, useful `4`, risk `9`, Boundary IoU `-0.000147`
- risk<=20: accepted `27`, useful `11`, risk `16`, Boundary IoU `+0.000052`

Decision:

- R216-R218 do not support mask-level promotion.
- Do not apply them to clean-test-v2.
- Do not claim R201 metric improvement from these candidate-action diagnostics.

## R219 Visual Audit

Artifacts:

- Clean-test diagnostic pack:
  `outputs/analysis/r206_hardcase_visualizations/index.html`
- Train/val action diagnostic pack:
  `outputs/analysis/r219_action_hardcase_visualizations/index.html`

R219 action pack:

- total panels: `24`
- safe useful: `8`
- hard risk / over-erosion: `8`
- ambiguous: `8`

Interpretation:

- Useful seam actions visibly exist.
- Hard-risk and ambiguous actions often look locally seam-like, explaining why
  simple geometry, hand-crafted patch features, and small crop CNNs fail to
  select actions safely.
- The current operation is still deletion-like; even partial deletion can
  remove true bone foreground.

## Claim Audit

### Can we currently claim improved mask-level anatomical consistency?

No.

Reason:

- No R216/R217/R218 mask-level validation result exists.
- Candidate diagnostics are not R201 mask-level evidence.
- Gate frontiers are too weak and risky.
- clean-test-v2 has not been used for these branches, correctly.

### Can we claim a useful discovered signal?

Yes, but only as internal experimental evidence:

- partial seam/background actions can improve boundary/gap diagnostics under
  oracle/diagnostic grouping;
- hard deletion is worse than partial action;
- safe-channel diagnostics identify a real failure-mode signal.

### Can we continue the same gate-threshold route?

No.

R216, R217, and R218 all failed to produce a reliable grouped-CV low-risk
frontier. More threshold sweeps or another shallow classifier would likely
repeat the same failure.

## Next Route

Recommended R221 direction:

Instance-preserving seam/background formulation.

Core idea:

- stop treating the action as foreground deletion;
- predict or infer a seam/background protection map;
- only suppress foreground where a seam/background channel is supported;
- enforce that component count error and recall do not worsen.

Possible implementation stages:

1. Train/val-only feasibility audit:
   - use GT only to compute an oracle seam/background suppression upper bound;
   - require component count MAE non-worsening;
   - require Dice/IoU/Recall non-degradation;
   - measure boundary/gap gains.
2. If feasible, train a GT-free seam/background scorer:
   - input: image, R110 anchor, boundary/distance maps;
   - output: seam/background probability or protection map;
   - loss: foreground preservation + gap/seam emphasis.
3. Apply only on original val first:
   - write isolated masks;
   - run R201-style metrics on val;
   - visualize hard cases.
4. Only after validation lock, apply once to clean-test-v2.

Parallel non-model work:

- complete representative baseline comparison, especially Swin-UNet /
  SwinUNETR / TransUNet style baselines if feasible;
- keep nnU-Net as risk audit, not the primary Dice target.

## Stop Rules

Do not promote a new branch unless it has original-val evidence that:

- Dice/IoU/Recall do not materially degrade;
- Boundary IoU/F1 improve meaningfully;
- gap-region FP improves;
- component merge/count error do not worsen;
- visual audit does not show over-erosion.

Do not use clean-test-v2 for threshold search, model selection, or deciding
which branch to keep.

