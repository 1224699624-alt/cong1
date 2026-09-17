# R316C Dual-Route Hard-Gate Plan

Status: implementation approved by user; ablation design pending external reviewer confirmation.

## Scope

- Dataset: `TSRS_RSNA-Epiphysis` only, full 875 train / 96 original-val.
- No `TSRS_RSNA-Articular-Surface` data.
- No test or clean-test-v2 access.
- Evaluation: frozen R201 definitions on the same 96 original-val cases.
- Mature initialization: exact R202 checkpoint used by R260.

## Reference and hard gate

R260 is the reference to beat:

- Dice: `0.9027585385931185`
- IoU: `0.826195708970261`
- HD95: `15.864083649434889 px`
- ASSD: `4.320394541478605 px`

A candidate passes only if all four conditions hold:

1. Dice `> 0.9027585385931185`.
2. IoU `> 0.826195708970261`.
3. HD95 `<= 15.070879466963144` (at least 5% lower than R260).
4. ASSD `<= 4.104374814404675` (at least 5% lower than R260).

Recall is reported but is not a selection gate. Boundary IoU/F1, Surface Dice,
gap-region FP, merge rate, and component-count MAE remain secondary diagnostics.

## Paired routes

### R316C-A: continuous R260 prior

- Retrain the R258B continuous relation prior on full 875/96.
- Pair input: native-detail 256 crop plus the same crop downsampled to 128 and
  resized back as a global-context channel.
- Center maps use `sigma = clip(0.15 * local_scale, 3, 12)` instead of fixed 5.
- Small training-only center perturbation improves tolerance to proposal error.
- Full-image prior remains continuous; no binary relation threshold is added.

### R316C-B: safe R316B prior

- Retrain the R316B hard-negative/support selector with the same 256/128 input,
  adaptive center sigma, and training-only center perturbation.
- Keep the audited relation threshold 0.65 so the route differs from A only in
  its safe-selector semantics, not a newly searched threshold.
- Preserve the predicted two-sided support suppression.

## Loss-weight ablation

Each route uses the exact R260 nnU-Net initialization and tests:

- `alpha=0.020`: conservative, intended to protect Dice/IoU.
- `alpha=0.035`: stronger interface correction, intended to retain HD95/ASSD gains.

The old `alpha=0.050` results are R260 and R319 and are not rerun.

## Selection

Candidates failing either Dice or IoU are rejected even if HD95/ASSD improve.
Among candidates passing all hard gates, select the largest mean relative
reduction of HD95 and ASSD; use Dice as the tie-breaker.

## Compute estimate

- Prior retraining and map generation: approximately 0.5-1.5 GPU hours.
- Four mature nnU-Net continuations: approximately 3-4 GPU hours on RTX 4090D.
- Inference, R201, and visualizations: approximately 0.5 GPU hours.

