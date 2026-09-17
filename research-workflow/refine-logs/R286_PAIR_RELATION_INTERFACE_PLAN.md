# R286 Pair-Relation Local-Interface Prior Plan

## Purpose

Upgrade R285's global one-channel overlap map to a pair-specific and relation-conditioned
local-interface prior without changing the official RAM-W600 split or threshold protocol.

## Fixed protocol

- Dataset: RAM-W600 BoneSegmentation only (`425/69/124`).
- Backbone and initialization: mature R285 nnU-Net baseline checkpoint.
- Selected pairs: top 15 overlap pairs measured on training labels only.
- Relations: `gap`, `overlap`, and `uncertain/support` for every selected pair.
- Interface ROI: intersection of radius-7 dilations of the two instance masks.
- Selection: validation macro-Dice gate followed by overlap Dice; the unchanged baseline
  identity is an explicit candidate.
- Fixed inference threshold: `0.5`; no test threshold search or test model selection.
- `clean-test-v2` is not used.

## Model changes

- Fuse the last two high-resolution nnU-Net decoder features.
- Predict `15 x 3` pair-relation maps rather than one global overlap map.
- Apply pair-specific signed corrections only to the two incident bone channels.
- Warm up the relation head with a frozen backbone, then unfreeze only the final two
  decoder stages at low learning rate.
- Distill the mature teacher outside all selected local interfaces.

## Gate

- Primary: test macro Dice must not materially decrease.
- Local: overlap Dice and overlap NSD 2 px should improve.
- Diagnostic: inspect per-pair Dice/NSD to ensure gains are not driven by a few pairs.
