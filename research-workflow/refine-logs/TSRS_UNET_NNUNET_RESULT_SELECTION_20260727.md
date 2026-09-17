# TSRS U-Net / nnU-Net result selection

Date: 2026-07-27

## Dataset decision

Use the manually reviewed clean-panel selection for the next paired U-Net and nnU-Net experiments:

- 814 retained training cases;
- 94 retained original-val cases;
- excluded validation cases: `2246`, `5882`;
- panel files define a case keep-list and are not model inputs;
- no label-pixel edits were detected;
- test and clean-test-v2 remain unused.

The decision is based on the same 94 validation cases. Clean continuation of the prior nnU-Net improves Dice, IoU, Recall, boundary metrics, gap FP, surface metrics, HD95 and ASSD relative to the pre-clean prior checkpoint; only component merge rate worsens by one case. Clean continuation also reduces the standard model's foreground leakage and merge rate, although it slightly lowers Dice and Recall.

## nnU-Net: complete paired evidence

| Metric | Clean standard R310 | Clean prior R308 | Prior - standard |
|---|---:|---:|---:|
| Dice | 0.907059 | 0.906858 | -0.000201 |
| IoU | 0.832534 | 0.832050 | -0.000483 |
| Precision | 0.911726 | 0.917879 | +0.006153 |
| Recall | 0.906142 | 0.899361 | -0.006782 |
| Boundary IoU | 0.247588 | 0.246876 | -0.000712 |
| Boundary F1 | 0.389773 | 0.388826 | -0.000948 |
| gap-region FP rate | 0.186640 | 0.173044 | -0.013597 |
| component merge rate | 0.595745 | 0.521277 | -0.074468 |
| component count MAE | 1.840426 | 1.734043 | -0.106383 |
| Surface Dice 2 px | 0.671691 | 0.671301 | -0.000390 |
| Surface Dice 5 px | 0.914824 | 0.917256 | +0.002431 |
| HD95 px | 10.281937 | 7.827458 | -2.454479 |
| ASSD px | 2.955085 | 2.571234 | -0.383851 |

The clean prior version is better for separation, merge control and large surface errors, but its foreground suppression remains too strong. It is the current nnU-Net improved-system checkpoint, not yet a final positive result.

## U-Net: current evidence is incomplete

R272 is a compact U-Net trained on the uncleaned 875/96 split with instance-derived seam supervision and separation/support losses. Its selected epoch reports:

| Metric | R272 prior + loss U-Net |
|---|---:|
| Dice | 0.852395 |
| IoU | 0.750940 |
| Precision | 0.777943 |
| Recall | 0.953245 |
| Boundary IoU | 0.388504 |
| component count error | 2.843750 |
| false bridge flag | 0.718750 |
| seam IoU | 0.131098 |

These values cannot be used as a baseline-vs-method improvement claim because:

1. there is no same-architecture plain U-Net arm;
2. it uses the uncleaned split;
3. its metrics are computed at 512 x 512 with the pilot evaluator rather than the complete original-resolution R201 protocol.

## Required final two-backbone table

Run both U-Net arms on the selected clean 814/94 split:

1. plain U-Net with ordinary Dice/BCE;
2. the identical U-Net initialization plus the frozen prior input and separation/support loss.

Use the same seed, augmentation, checkpoint rule, inference threshold and original-resolution R201 evaluation. The already completed R310/R308 pair supplies the nnU-Net row. Until the clean paired U-Net run exists, the paper-ready two-backbone result table has one complete backbone, not two.

