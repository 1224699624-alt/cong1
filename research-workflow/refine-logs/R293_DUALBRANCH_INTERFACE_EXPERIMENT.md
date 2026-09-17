# R293 Explicit Dual-Branch Interface Experiment

Date: 2026-07-23

## Scope and protocol

- Dataset: `TSRS_RSNA-Epiphysis` train plus original-val only.
- `TSRS_RSNA-Articular-Surface` is excluded.
- `clean-test-v2` is not read, tuned on, or evaluated.
- Baseline checkpoint: mature R275 control, SHA256
  `557f78d7bd5f2987c6e224b87461f939b006df7822ab3170167e0502d9092c27`.
- Final development audit: the existing R201 single-split evaluator on all 96
  original-val images.
- Raw data are unchanged; all R293 artifacts are isolated.

## Paired design

### R293-A: explicit dual-branch control

- Frozen mature R275 nnU-Net is the base predictor.
- A quarter-resolution global branch models full-patch layout.
- A native-resolution local branch consumes X-ray, image gradient, and
  high-pass evidence.
- An image/prediction-driven soft interface gate restricts local residuals.
- Global and local residual heads initialize at exact zero, so the initial
  prediction is exactly R275.
- No gap, support, age, sex, relation-type, fixed-pair, or GT-ROI objective.

### R293-B: dual branch plus interface prior/loss

- Independently initialized from the same R275 checkpoint, not R293-A.
- Online wide gap-band is generated after augmentation from background pixels
  bracketed by foreground in opposing directions.
- Gap loss is a logit-space hinge requiring foreground probability <= 0.20.
- Support loss is a logit-space hinge requiring foreground probability >= 0.75.
- Support is a local bone-interior ring separated from the gap by a four-pixel
  safety buffer.
- The local separation-prior map is supervised as high on gap and low on
  support.
- Inference takes only the X-ray; no label or GT ROI enters the network.

## Decision gate

R293-B is promotable only if it improves the interface/topology profile over
both R275 and R293-A without a material Dice/Recall loss. Primary diagnostics:

- gap-region FP rate (lower)
- component merge rate (lower)
- component count MAE (lower)
- Boundary IoU/F1 (higher)
- Surface Dice 2 px / 5 px (higher)
- HD95 and ASSD (lower)
- Dice and Recall must remain clinically/experimentally non-inferior

## Remote run

- Host: `connect.westc.seetacloud.com:43758`
- Workspace: `/root/autodl-tmp/YOLO_SAM_generic_src`
- GPU: RTX 4090 D, GPU 0
- Screen: `r293_dualbranch`
- Launcher: `run_r293_dualbranch_interface_nnunet.sh`
- Log: `outputs/bridge_logs/r293_dualbranch_interface_nnunet.log`
- Status at launch: dataset materialization/preprocessing in progress.

## Completed result

The paired chain completed at 2026-07-23 17:04 +08:00. R293-A stopped after
25 epochs; R293-B stopped after 12 epochs. All 96 original-val cases were
predicted and evaluated.

| Metric | R275 | R293-A | R293-B | B minus R275 |
| --- | ---: | ---: | ---: | ---: |
| Dice | 0.897965 | 0.891461 | 0.890785 | -0.007180 |
| Recall | 0.899057 | 0.887206 | 0.881381 | -0.017677 |
| Precision | 0.902028 | 0.904651 | 0.909475 | +0.007447 |
| Boundary IoU | 0.234335 | 0.231354 | 0.230721 | -0.003614 |
| Boundary F1 | 0.373105 | 0.368992 | 0.368080 | -0.005024 |
| gap-region FP | 0.187305 | 0.178589 | 0.167867 | -0.019438 |
| component merge rate | 0.510417 | 0.489583 | 0.489583 | -0.020833 |
| component count MAE | 2.333333 | 2.114583 | 2.125000 | -0.208333 |
| Surface Dice 2 px | 0.650056 | 0.643377 | 0.642345 | -0.007711 |
| Surface Dice 5 px | 0.900422 | 0.896792 | 0.896651 | -0.003771 |
| HD95 | 16.350589 | 20.000352 | 19.987032 | +3.636443 |
| ASSD | 6.165032 | 5.489731 | 5.331499 | -0.833533 |

Decision: `no-go` as a promoted model, but positive mechanism evidence.

- R293-A already creates most of the Dice/Recall loss, so the explicit global
  residual branch is the main source of region degradation.
- Relative to R293-A, R293-B further reduces gap FP by 0.010722 and raises
  precision by 0.004823, but lowers recall by 0.005825 and does not further
  reduce merge rate.
- Thus the loss suppresses foreground near gaps but does not yet convert that
  suppression into more fully separated components.
- The R293-B gap hinge decreased from 0.775942 to 0.604167, proving the
  non-saturating objective is trainable. However, its gradient also reached the
  unrestricted global residual head; that head grew to absmax 0.061365 versus
  0.020248 in R293-A. This explains the excessive conservative/global shift.
- The severe R292-like failures remain (`1886`, `1518`, `2709`), dominating
  HD95 and showing that the global residual is not sufficiently locality-safe.

Recommended next design change (not a scalar weight sweep): prevent interface
loss gradients from updating the global branch, bound the global correction,
and route the separation objective only through a sparse local correction
inside the prior gate. Keep the R275 identity path exact outside that gate.
