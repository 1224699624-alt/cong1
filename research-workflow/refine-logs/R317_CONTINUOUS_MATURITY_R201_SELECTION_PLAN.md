# R317 Continuous-Prior Maturity and R201 Checkpoint Selection

## Scope

- Dataset: `TSRS_RSNA-Epiphysis` only.
- Full split: 875 train / 96 original-val.
- No Articular-Surface, test, or clean-test-v2 access.
- Mature initialization: exact R202 checkpoint used by R260/R316c.
- Baseline: R260 under frozen R201 definitions.

## Question

R316c passed its hard gate with the continuous prior at `alpha=0.035`, but the
relation prior was trained for only six epochs and nnU-Net selected its best
checkpoint using EMA foreground Dice. R317 tests whether a converged prior and
R201-aligned checkpoint selection retain or improve that result.

## Prior maturity protocol

- Preserve the R316c 256/128 pair representation, adaptive Gaussian, proposals,
  metadata, loss, optimizer, and three fixed seeds.
- Extend training from 6 to 18 epochs.
- Save every prior epoch.
- For each seed, require close-pair recall >= 0.75, contact recall >= 0.70,
  negative specificity >= 0.75, and heat coverage >= 0.85; among eligible
  epochs select maximal heatmap soft Dice, then AUROC/AP/specificity.
- Generate full 875/96 continuous maps from the three selected conditioned
  checkpoints. No GT is used during map inference.

## Segmentation checkpoint protocol

- Train only the successful continuous `alpha=0.035` route.
- Keep the R260 training recipe and EMA Dice early-stop safety guard.
- Save a checkpoint every three completed epochs plus nnU-Net's EMA-best.
- Run full 96-image original-val inference and R201 for every saved checkpoint.

## Hard-gate selection

A checkpoint is eligible only if all conditions hold against R260:

1. Dice > 0.9027585385931185.
2. IoU > 0.826195708970261.
3. HD95 <= 15.070879466963143 (at least 5% better).
4. ASSD <= 4.104374814404675 (at least 5% better).

Among eligible checkpoints, select maximum mean relative HD95/ASSD reduction,
using Dice only as a tie-breaker. Boundary, surface, gap, and component metrics
remain diagnostics. This selection uses original-val only; clean-test-v2 stays
locked.
