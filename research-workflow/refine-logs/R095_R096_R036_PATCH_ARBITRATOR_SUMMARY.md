# R095/R096 R036 Patch Arbitrator Summary

## Objective

Test whether the R090-style patch/CNN arbitrator can exploit the older high-recall R036 MedSAM+GBC candidate. R037 had shown a GT-leaking R025b+R036 pixel oracle Dice of `0.941744`, but earlier R036 exploitation used weaker MLP/component/refiner routes.

## Runs

| Run | Status | clean-test-v2 Dice | Notes |
| --- | --- | ---: | --- |
| R095 | STOPPED_SPEED | n/a | full local_stats patch features ran at about `20-25 s/batch`; stopped before epoch 1 completed |
| R095b | DONE_NEW_BEST_TINY | `0.916641` | basic-feature fast patch arbitrator, new valid best by only `+0.000201` over R090 |
| R096 | DONE_PATH_BLOCK | pixel oracle `0.924049` | R038/R090/R095b oracle remains below target |

## R095b Metrics

| Metric | Value |
| --- | ---: |
| Dice | `0.916641014890419` |
| IoU | `0.8466608561896706` |
| Precision | `0.9002255526090882` |
| Recall | `0.9349997044164438` |
| Boundary IoU | `0.24555864844426387` |

Compared with R090 Dice `0.9164396820551889`, R095b improves by only `+0.0002013328352301`. It remains `0.0151249917653235` below the target `0.9317660066557425`.

## R096 Oracle Audit

| System / Combination | Dice |
| --- | ---: |
| R038 | `0.914357` |
| R090 | `0.916440` |
| R095b | `0.916641` |
| intersection_all / vote_ge_3 | `0.917075` |
| per-image oracle | `0.917515` |
| GT-leaking pixel oracle | `0.924049` |

Even the GT-leaking pixel oracle is below target by `-0.007717`. This closes the R036+R090/R095b patch/fusion route for the revised +0.02 target.

## Decision

Do not continue:

- R036 threshold sweeps;
- R036 local_stats patch reruns;
- R038/R090/R095b voting or learned fusion;
- heavier patch arbitration over the same candidate masks.

Next branch must introduce a genuinely new image-model source. R097 starts that branch with a `timm` ConvNeXt-tiny encoder plus U-Net/FPN decoder direct segmenter.
