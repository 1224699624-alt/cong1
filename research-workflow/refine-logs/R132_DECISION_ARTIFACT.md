# R132 Decision Artifact

## Goal

Target remains clean-test-v2 Dice `> 0.9317660066557425`.
Success is evaluated only on `TSRS_RSNA-Epiphysis_clean_test_v2/test`; the new reannotated test split is excluded.

## Current State

The valid best is still R110:

- clean-test-v2 Dice `0.9177231563529792`
- target gap `0.01404285030276331`

Recent branches are closed:

- R129 lightweight set/query instance segmentation failed with severe high-recall / low-precision overmasking.
- R130 DINOv3 bridge-suppressed instance-separation improved validation structure but reached only clean-test-v2 Dice `0.890938`.
- R131 showed R130 is not useful for per-image selection or non-leaking voting; R130 was selected in `0/81` per-image oracle cases.

## Chosen Mechanism

R132 should test a stronger query/no-object decoder without a full Mask2Former rewrite:

- Reuse the verified timm DINOv3 ConvNeXt-tiny FPN feature path from R117/R130.
- Replace the single binary source head with `K` query mask logits plus query objectness logits.
- Train query masks by Hungarian matching against instance-valued labels.
- Add explicit object/no-object calibration so unused queries are suppressed.
- Keep a binary union auxiliary head to stabilize foreground recall.
- Select only threshold/blend on original validation, then evaluate once on clean-test-v2 and original-test.

This differs from R129 by using a pretrained DINOv3/FPN backbone and explicit no-object calibration. It differs from R130 by predicting variable query masks rather than one bridge-prone binary foreground field.

## Gate

Run a tiny smoke first. Full R132 is justified only if smoke completes train -> val -> clean-test-v2 -> original-test without path errors.

Stop this branch if full R132 does not beat R110 on clean-test-v2. Do not follow with blind query-count or threshold sweeps unless validation and clean-test-v2 both show a clear gain.
