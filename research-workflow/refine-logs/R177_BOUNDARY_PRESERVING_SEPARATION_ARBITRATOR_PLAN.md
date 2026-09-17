# R177 Boundary-Preserving Separation Arbitrator Plan

**Date**: 2026-07-03

**Parent evidence**: R176 mask-synced audit.

## Motivation

R176 shows that R110 is the best completed system for Dice, Boundary IoU, Boundary F1, and separation-band foreground FP. However, it is still weak on under-separation:

- R110 false-bridge rate: `0.790123`
- R110 component-count error: `2.506173`

R130/R165 reduce component error but lose too much Dice and boundary quality. Therefore, the next model should not replace R110 with a new full-image source. It should keep R110 as the anchor and only edit local bridge/boundary risk zones.

## Train/Val Anchor Availability

Remote check on `/home/shenzeyu/workspace/YOLO_SAM_generic_src`:

| Experiment | train masks | val masks | Use |
| --- | ---: | ---: | --- |
| `r110_r100_r108_patch_basic` | missing | missing | apply-side clean-test-v2 anchor only |
| `r100_r095b_r097_patch_basic` | missing | missing | apply-side clean-test-v2 predecessor only |
| `r100_like_r025b_r097_patch_basic_trainval` | `875` | `96` | train/val anchor proxy |
| `r095b_r036_patch_basic` | missing | missing | not usable for R177 train/val |
| `r090_online_patch_disagreement_arbitrator` | missing | missing | not usable for R177 train/val |

Decision: R177 should use `r100_like_r025b_r097_patch_basic_trainval` as the train/val anchor proxy, then apply the locked rule to R110 clean-test-v2 masks. This is not perfect, but it is the available non-leaking proxy and matches the earlier R100/R110 patch-family distribution strategy.

## Candidate Mechanism

R177 should be a small local edit model, not a full segmenter.

Inputs per local patch:

- grayscale X-ray patch;
- anchor mask patch;
- anchor boundary band;
- anchor connected-component map proxy or distance-to-component-edge;
- GT-derived train-time separation band;
- optional R130/R165 source mask patch as structural hint, if available on train/val without leakage.

Outputs:

- foreground delete probability inside bridge-risk / separation-band zones;
- optional foreground add probability inside GT boundary-risk zones.

Risk zone:

- train/val: derived from anchor errors and GT components;
- apply: derived only from R110 component geometry and image/boundary features, not GT.

## R177 Gate Before GPU Training

R177 must pass a non-GPU/CPU preparation gate first:

1. Sync or verify train/val anchor proxy masks locally/remote.
2. Build `r177_trainval_bridge_risk_manifest.json`.
3. Confirm enough non-empty risk-zone pixels:
   - at least `50` train images with bridge-risk pixels;
   - at least `10` val images with bridge-risk pixels.
4. Confirm positive/negative class balance is not degenerate.
5. Run a tiny CPU smoke over a few images to write risk panels and manifest summary.

Only after this gate should a remote one-GPU training run be launched.

## Selection Rule

Select on original validation only:

- primary: Dice must not drop versus the train/val anchor proxy;
- secondary: Boundary F1 and Boundary IoU improve;
- structural: component-count error and false-bridge proxy improve;
- guardrail: separation-band FP does not regress.

Clean-test-v2 is evaluated once after the validation protocol is locked.

## Expected Outcomes

### Positive

If R177 improves R110 Dice or keeps Dice flat while reducing false-bridge/component error, it becomes the first branch that supports the user’s bone-gap adhesion claim.

### Mixed

If it improves false-bridge but loses Dice, the method can still support an auxiliary “structure-quality tradeoff” table, but not the main target.

### Negative

If risk zones are sparse or validation improvements do not transfer, close R177 and return to either:

- R168/R170 human-reviewed label protocol path;
- CUDA-toolkit-capable faithful MaskDINO/Mask2Former environment;
- a new external architecture with strict train/val smoke gates.

## Immediate Implementation Task

Create R177 preparation script:

`scripts/prepare_r177_bridge_risk_manifest.py`

The script should:

- read train/val GT labels;
- read `r100_like_r025b_r097_patch_basic_trainval` masks;
- compute component-count mismatch, anchor/GT boundary bands, and separation-risk bands;
- write summary JSON and a CSV manifest;
- optionally create a small panel pack for visual inspection.
