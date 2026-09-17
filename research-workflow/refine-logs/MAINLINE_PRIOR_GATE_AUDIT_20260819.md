# Mainline Prior/Gate Audit (2026-08-19)

## Locked Mainline

The project mainline is **not** a new prior search. It is:

```text
validated seam prior + validated overlap prior + conditional/gated loss
```

The current task is to redesign only the conditional loss so that the two
validated priors are complementary rather than mutually destructive.

The priors are dataset-specific because the label semantics differ:

| Dataset | Validated prior asset | Primary objective | Immutable anchor |
|---|---|---|---|
| TSRS_RSNA-Epiphysis | image/center-derived continuous seam prior | preserve/open background bone seam | R317 |
| RAM-W600 | native-resolution per-instance overlap/completion prior | recover missing memberships in true projected overlap | R329-R332 |

TSRS color IDs are not a reliable projection-overlap label. RAM's 14-channel
instance labels are reliable overlap supervision. A unified framework may share
the state-conditioned loss interface, but it must not force identical state
definitions or identical loss terms onto both datasets.

## Experiment Ledger and Decisions

### TSRS prior and loss line

| Runs | What was tested | Evidence | Decision |
|---|---|---|---|
| R260, R265-R272 | mature prior, anatomical/ROI/relation/seam variants | prior changes separation behavior, but targeting and contraction risks appear | retain prior concept; do not repeat generic ROI/seam variants |
| R279 | instance seam prior + one-sided suppression | gap FP `-0.01194`, merge `-0.04167`, but Dice `-0.00596`, Recall `-0.01276`, Boundary/Surface worse | prior retained; one-sided suppression rejected |
| R280 | seam suppression + broad bone support, equal weight | Recall `+0.01949`, but gap FP `+0.07985`, merge `+0.04167`, boundary collapse | reject broad two-sided support |
| R281 | lower support coefficient | still gap/merge and boundary failure | do not run manual support sweep |
| R282 | GradNorm seam/support | merge improved, but gap FP and Dice gate failed | scalar gradient balancing insufficient |
| R283 | spatial/confidence-gated support | support/seam overlap `0`, gap FP and merge improved, but Recall/Dice/HD95 failed | support placement helps leakage but support family rejected |
| R317 | continuous image-centers-only prior, mature native nnU-Net | current TSRS anchor; improves Dice, Boundary, Surface, HD95/ASSD, gap FP and merge on original-val | immutable seam-prior anchor |
| R333-R346/R348 | combined interaction losses and corrected seam-preserve attempts | R346 changed FP/FN tradeoff; R348 is a loss candidate only, not evidence that R317 prior is bad | audit loss; do not alter R317 prior without new prior evidence |

### RAM prior and loss line

| Runs | What was tested | Evidence | Decision |
|---|---|---|---|
| R284b | explicit overlap prior on compact U-Net | large overlap gains | establishes overlap prior can work on weak backbone |
| R285 | single overlap map/residual on strong nnU-Net | small negative deltas | reject global single-map residual |
| R286b-R288 | pair-specific local interface/boundary/directional gate | tiny overlap gains with macro preservation; decoder unfreezing degrades | retain pair-local/frozen-backbone principle |
| R324 | overlap lambda sweep | balanced region near `lambda=0.002`; stronger values over-constrain | no more blind lambda sweeps |
| R326 | layer-projection prior | mechanism-positive vs frozen baseline, not better than capacity control | do not replace validated overlap prior |
| R327-R329 | native-resolution per-instance completion | R329 beats baseline and generic adapter; corruption mismatch remains | retain instance-completion representation |
| R330-R331 | surface/volume refinement, five-seed validation | stable positive RAM validation trend | strong overlap anchor |
| R332 | joint iterative completion, two steps selected on validation | positive locked test vs R325 on area, overlap and surface metrics; two adverse metrics remain | immutable RAM anchor |
| R334-R347 | dual interaction, fixed support/gate, corrected loss | R334/R335/R346/R347 all fail; R347 best is epoch 0 and later epochs collapse | reject current training recipe, not R332 prior |

## Failed Families That Must Not Be Repeated

1. Global or broad two-sided bone support on TSRS.
2. Manual support coefficient sweeps after R281.
3. Scalar GradNorm as the sole conflict resolver after R282.
4. GT-only state selection combined with GT-free inference.
5. Global single overlap-map residual on strong nnU-Net.
6. Unfreezing a mature decoder together with a new interaction loss.
7. Shared residual training on selected GT-hard instances followed by all-instance inference.
8. Teacher preservation applied as a large, unconditional penalty from the first update.

## What Is Still Unresolved

The unresolved question is narrow and testable:

> Can a state-conditioned loss route the already validated seam and overlap
> priors to disjoint, dataset-appropriate regions while preserving the mature
> R317/R332 outputs outside the active interaction region?

This is a loss-routing question, not a prior-construction question.

## Correct Next Design

The next candidate must satisfy all of the following before full training:

### Shared loss interface

```text
L = L_base
  + g_sep(x) * lambda_sep * L_sep
  + g_ov(x)  * lambda_ov  * L_ov
  + g_pres(x) * lambda_pres * L_pres
```

- `g_sep` and `g_ov` are mutually exclusive;
- gates are generated from prior/model outputs available at inference;
- no GT-selected instance subset is used in training;
- `L_pres` is a non-interference term, not a new structural prior;
- every term is normalized by its active pixel/instance mass and logged by
  gradient norm, not only scalar loss magnitude.

### TSRS adapter

- `g_sep`: high-confidence R317 seam corridor, restricted to background;
- `L_sep`: seam/background separation only;
- `g_ov = 0`: TSRS does not have trustworthy projection-overlap labels;
- `g_pres`: outside the seam corridor, preserve the full R317 class
  distribution with a low, ramped weight;
- do not add bone-support restoration unless a new prior audit proves R317 is
  inadequate.

### RAM adapter

- `g_ov`: predicted pair-intersection/interface state from the existing R332
  overlap prior, not GT-only masks;
- `L_ov`: pair-local instance completion/boundary objective;
- `g_sep = 0` for the TSRS seam state; RAM's primary state is overlap;
- preserve R332 step-2 output outside active pair interfaces;
- initialize residual to zero and include epoch-0 as a mandatory checkpoint;
- freeze mature backbone and train only the smallest validated interaction head
  first.

## Required Validation Order

1. Pure routing smoke: finite loss, nonzero gradients, zero gate overlap.
2. Identity audit: epoch-0 output exactly equals R317/R332 anchor.
3. Train/validation path audit: same instance count and same gate source.
4. 2-4 image overfit: each active loss must reduce its own local error without
   increasing the inactive-region error.
5. Original validation only; no TSRS clean-test-v2 or RAM test use.
6. Hard gates against the matched anchor; reject any candidate that trades
   decisive local metrics for a small global Dice gain.
7. Only after validation passes: multi-seed confirmation, then locked test.

## Current Run Status

R347 RAM and R348 TSRS are historical/diagnostic candidates. Their outputs must
remain isolated and must not redefine the anchors. R317 and R332 remain the
mainline prior assets until a loss-only candidate passes the above gates.
