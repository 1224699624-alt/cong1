# R349-R350 Latest Results Summary (2026-08-24)

> 2026-09-17 correction: this historical narrative contains superseded comparisons. TSRS historical native HD95=54.34/ASSD=14.17 is not a matched comparator; use the R350 matched recheck (16.9633/4.3815). R350 RAM's selected epoch29 has Pair MSD=1.6543578856 and legacy Seam FP=0.0101082319; older text mixes epoch30 values. The legacy Seam FP uses prediction-dependent regions and is not a fixed-ROI comparison. See `../status-audit-20260917/PROJECT_STATUS.md` and the corrected `outputs/analysis/latest_best_vs_baseline_20260824.csv`; the original CSV was archived before correction.

## Protocol boundary

- TSRS tuning/evaluation uses `original-val`; `clean-test-v2` remains locked.
- RAM R349/R350 model selection uses validation; RAM test remains locked.
- R349 TSRS has only nnU-Net validation Dice/IoU and cannot be compared as a full R201 result.
- R350 TSRS has a full R201 audit.

## Baseline correction (2026-08-24)

A fresh native nnU-Net inference was run through the exact Dataset204 input,
prediction conversion, 96-case original-val set, and R201 evaluator used by
R350. It produced Dice `0.903663`, IoU `0.827698`, HD95 `16.963342 px`, and
ASSD `4.381480 px`. Consequently, the historical R260 baseline values Dice
`0.897112`, IoU `0.819215`, HD95 `54.3414 px`, and ASSD `14.1738 px` must not
be used as the matched R350 baseline. They came from preserved R202 masks made
through an earlier inference/preprocessing path.

Against the corrected matched baseline, R350-best is lower by `0.00001754`
Dice and `0.00003342` IoU, while improving all registered boundary, surface,
gap, merge, and component-count metrics. Thus R350 is structurally positive
but fails the strict Dice/IoU-above-baseline rule by a very small margin.

## Current best established anchors

- TSRS: R317 continuous seam prior/loss initialized from mature nnU-Net.
- RAM: R332 two-step iterative overlap completion initialized from R325 native nnU-Net.

## Latest unified-loss experiments

### R349

- TSRS: Dice +0.000905 and IoU +0.001600 versus matched R317 nnU-Net validation summary, but no full R201 audit. Its overlap state was disabled, so it was not a truly isomorphic two-state experiment.
- RAM: epoch 28 passed the validation gate. All registered overlap/pair metrics improved slightly over the R332 validation anchor; Seam FP decreased by 0.001108.

### R350

- Uses one shared three-state loss interface and parameter-space PCGrad for both datasets.
- TSRS: Dice/IoU, Boundary IoU/F1 and Surface Dice 2 px improved slightly versus R317, but Gap FP, merge rate, component-count MAE, Surface Dice 5 px, HD95 and ASSD worsened. Overall decision: no-go versus R317.
- RAM: epoch 29 passed every registered gate. Overall, overlap, pair and seam metrics all moved in the favorable direction versus the matched R332 validation anchor. Overall decision: effective but small incremental gain.

## Claim supported by current evidence

R350 supports a narrow claim: state-conditioned loss with parameter-space conflict routing can add a small, non-destructive gain on RAM's reliable multi-instance overlap representation. It does not yet support the claim that the same state definitions solve TSRS separation and RAM overlap equally well.

## Best-versus-baseline table

The machine-readable comparison is stored in `outputs/analysis/latest_best_vs_baseline_20260824.csv`.
