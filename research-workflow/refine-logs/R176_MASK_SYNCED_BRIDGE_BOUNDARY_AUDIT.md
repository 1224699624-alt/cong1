# R176 Mask-Synced Bridge/Boundary Audit

**Date**: 2026-07-03

**Goal**: Fairly recompute bridge, boundary, and component metrics from synchronized prediction masks on `TSRS_RSNA-Epiphysis_clean_test_v2/test`.

**Success-evaluation rule**: clean-test-v2 is used only for locked final/diagnostic evaluation. New reannotated test is not used.

## Artifacts

- Sync helper: `scripts/sync_r176_bridge_boundary_masks.py`
- Synced mask summary: `outputs/analysis/r176_mask_sync_summary.json`
- Full audit JSON: `outputs/analysis/r176_mask_synced_bridge_boundary_audit.json`
- Metric script: `scripts/audit_r175_bridge_boundary_claim_metrics.py`

Synchronized from the remote server:

- R090 `r090_online_patch_disagreement_arbitrator`
- R100 `r100_r095b_r097_patch_basic`
- R110 `r110_r100_r108_patch_basic`
- R130 `r130_dinov3_bridge_suppressed_instance_sep`
- R143 `r143_highres_medical_recipe_unet`
- R165 `r165_filtered_reannotated_dinov3_instance_sep`

Each experiment has `81/81` clean-test-v2 masks locally. The clean-test-v2 `test_labels` were also synced locally for metric computation only.

## Results

| System | Dice | Boundary IoU | Boundary F1 | Sep-band FP | False bridge | Component error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R090 patch arbitrator | `0.916440` | `0.251366` | `0.394480` | `0.213006` | `0.074074` | `9.345679` |
| R100 patch complement | `0.917618` | `0.251759` | `0.394545` | `0.202583` | `0.790123` | `2.506173` |
| R110 current best | `0.917723` | `0.251896` | `0.394721` | `0.201358` | `0.790123` | `2.506173` |
| R130 bridge penalty source | `0.890938` | `0.180672` | `0.301228` | `0.317425` | `0.641975` | `1.543210` |
| R143 high-res medical recipe | `0.891375` | `0.161392` | `0.271987` | `0.450644` | `0.888889` | `3.913580` |
| R165 filtered reannotation source | `0.890303` | `0.184990` | `0.307025` | `0.252102` | `0.641975` | `1.666667` |

## Interpretation

R110 forms a defensible advantage on:

- total region accuracy: best Dice and IoU;
- contour quality: best Boundary IoU and Boundary F1;
- local gap cleanliness: lowest separation-band foreground false-positive rate among the compared systems.

R110 does **not** form a defensible advantage on:

- false-bridge / under-segmentation rate;
- component-count error.

R090 has a low false-bridge rate because it tends to over-segment rather than merge, but its component error is very high. R130 and R165 reduce component error relative to R110, but their Dice, boundary metrics, and separation-band precision are much worse. Therefore, none of the completed branches is a clean SOTA-style answer to both contour quality and anatomical component consistency.

## Decision

Proceed to R177 rather than claiming that R110 already solves bone-gap adhesion.

The correct next mechanism is a **boundary-preserving separation arbitrator**:

- keep R110 as immutable anchor outside risk zones;
- identify train/val-defined bridge-risk zones where R110-like masks under-separate neighboring components;
- make only local delete/add decisions around separation and boundary bands;
- optimize for a three-way gate: no Dice drop, improved Boundary IoU/F1, reduced false-bridge/component error.

This is more aligned with the failure mode than another full-image source model, because prior source models improved one structural metric only by sacrificing Dice and boundary quality.

## R177 Gate

R177 should be trained and selected on original train/val or an isolated train/val variant only. Clean-test-v2 can be evaluated once after protocol lock.

Minimum launch requirements:

1. Materialize or sync R110-like train/val anchor masks.
2. Build train/val bridge-risk labels from GT component gaps and anchor under-segmentation.
3. Verify a CPU smoke on a small subset.
4. Launch one-GPU remote training only if smoke confirms non-empty risk zones and paired masks.

Success criteria:

- clean-test-v2 Dice `>= R110`;
- Boundary IoU and Boundary F1 improve over R110;
- false-bridge and component-count error improve over R110;
- separation-band FP rate does not regress.
