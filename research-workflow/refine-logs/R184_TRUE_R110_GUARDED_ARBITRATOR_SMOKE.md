# R184 True-R110 Guarded Arbitrator Smoke

R184 was run on the remote server only.

## Purpose

R177 used a proxy train/val anchor and failed by fragmenting masks. R183 materialized true R110 train/val masks, so R184 tested whether a stricter R110-anchored local arbitrator could make safe bone-gap / boundary repairs without destroying component topology.

## Setup

- Train/val anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 anchor: `r110_r100_r108_patch_basic`
- Training split: original `TSRS_RSNA-Epiphysis/train`
- Selection split: original `TSRS_RSNA-Epiphysis/val`
- Success evaluation: `TSRS_RSNA-Epiphysis_clean_test_v2/test`
- Server screen: `r184_smoke_gpu0`
- GPU: one RTX 4090, GPU0

The first risk-manifest attempt was too slow because full train/val morphology scanning is CPU-bound. The smoke launcher was narrowed to a sampled manifest for path and gate validation.

## Result

Smoke completed normally. Validation gate technically passed, but the selected configuration performed no edits.

- clean-test-v2 Dice: `0.9177231563529792`
- Boundary IoU: `0.25189601044085763`
- Component-count error: `2.506172839506173`
- False-bridge flag mean: `0.7901234567901234`
- Edited fraction: `0.0`
- Status: `below_best`

This exactly matches the R110 anchor and does not reduce the target gap.

## Decision

Do not launch the R184 full run. Under the current conservative guardrails, the local MLP degenerates to a no-op; relaxing it risks repeating R177 fragmentation. The next valid route must either:

1. require non-zero edits on validation while enforcing component-level acceptance after each image, or
2. move to a materially different model family / environment route rather than another local pixel-arbitrator sweep.
