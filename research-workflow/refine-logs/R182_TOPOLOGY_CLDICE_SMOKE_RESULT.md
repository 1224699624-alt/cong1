# R182 Topology/clDice Smoke Result

Date: 2026-07-03

## Purpose

R182 tested a new constraint route after R181 closed old candidate fusion and DINOv3 loss-weight sweeps. The idea was to add a topology-preserving soft-skeleton / clDice-style loss to the existing DINOv3 ConvNeXt-tiny instance-separation framework.

This was run on the server only. No local training was run.

## Implementation

Changed `scripts/train_timm_instance_separation_segmenter.py`:

- added disabled-by-default `--cldice-loss-weight`;
- added `--skeleton-iters`;
- added differentiable soft erosion/dilation/opening and soft skeleton functions;
- added clDice loss on the binary mask head when the weight is positive.

Default weight is `0.0`, so previous experiments are unchanged.

Launcher:

- `outputs/bridge_logs/run_r182_topology_cldice_smoke.sh`

Monitor:

- `scripts/monitor_r182_topology_cldice_smoke.py`

Artifacts:

- `outputs/bridge_logs/r182_topology_cldice_smoke.log`
- `outputs/timm_instance_sep/r182_topology_cldice_smoke/history.json`
- `outputs/analysis/r182_topology_cldice_smoke_clean_test_v2_metrics.json`
- `outputs/analysis/r182_topology_cldice_smoke_original_test_metrics.json`
- `outputs/analysis/r182_topology_cldice_smoke_gate_summary.json`

## Gate Setup

Smoke limits:

- train: `16` original train images
- val: `8` original val images
- clean-test-v2 diagnostic: `4` images only
- original-test control: `4` images only
- GPU: server GPU0

Important: clean-test-v2 was diagnostic only and was not used for tuning or success selection.

## Result

Gate status: `gate_failed`

Reason: `val_dice_too_low`

Best validation row:

- epoch: `7`
- val Dice: `0.364126`
- val Precision: `0.328357`
- val Recall: `0.448050`
- val Boundary IoU: `0.019221`
- val false-bridge flag: `1.000000`
- threshold: `0.55`
- sep suppress weight: `0.45`

Clean-test-v2 four-image diagnostic:

- Dice: `0.284047`
- Precision: `0.216344`
- Recall: `0.455470`
- Boundary IoU: `0.005653`
- component-count error: `22.000000`
- false-bridge flag: `1.000000`

## Interpretation

This branch is not ready for a full run. The topology/clDice loss did not provide a stable small-data learning signal in the current DINOv3 instance-separation script. It collapsed toward too few predicted components and poor boundary overlap.

This is different from R179:

- R179 learned topology better but lost too much Dice at full scale.
- R182 did not even pass the small smoke/overfit gate, so it should not be scaled.

## Decision

Do not launch full R182.

Do not run a blind clDice weight sweep in this framework.

The next useful route is still one of:

1. R168 reviewed CSV -> R172/R170/R171 server-only training.
2. A genuinely new architecture/environment route with a stronger overfit gate, not another small loss term on the same DINOv3 instance-separation framework.
3. A non-training analysis route that turns R176/R179 evidence into a defensible metric/claim package while being explicit that current best remains R110.
