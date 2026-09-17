# R068/R069 Instance-Separation Summary

**Date**: 2026-06-26

## Motivation

R065 compact context segmenter failed with clean-test-v2 Dice `0.889453` and high false-bridge rate `0.728395`. The key hypothesis for R068 was that R065 discarded instance-valued label information by binarizing masks, so a HoVer-inspired instance-separation model might reduce bridges and recover boundary structure.

## Results

| Run | Method | clean-test-v2 Dice | Precision | Recall | Boundary IoU | False Bridge | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| R038 | current best anchor | `0.914357` | `0.899054` | `0.931483` | `0.243176` | `0.296296` | baseline |
| R065 | compact context direct segmenter | `0.889453` | `0.836930` | `0.951533` | `0.175126` | `0.728395` | negative |
| R068 | HoVer-inspired direct segmenter | `0.903030` | `0.879287` | `0.929789` | `0.203925` | `0.580247` | mixed negative |
| R069 | R068 separation veto on R038 | `0.913944` | `0.900800` | `0.928770` | `0.242219` | `0.209877` | negative |

## Interpretation

Instance-separation supervision is a real signal: R068 improves over R065 on precision, component-count error, and false-bridge rate. R069 confirms that the learned separation head can reduce bridges when applied to R038.

But the gain is structurally useful rather than Dice-sufficient. R069 trims false bridges, but the recall loss lowers Dice below R038. A broader trimming grid is unlikely to close the `+0.017409` gap from R038 to the target because the best proxy-selected rule already shows the precision/recall tradeoff.

## Decision

Do not continue simple R038 trimming grids.

Next ARIS choice should be one of:

1. R070 shape/instance model with explicit recall recovery, not only separation trimming.
2. Data/label audit focused on clean-test-v2 cases where R038 remains below target and where all learned variants lose recall.
3. A hybrid instance proposal method that adds missing components while using separation only as a veto, rather than removing pixels from R038.

The target remains clean-test-v2 Dice `> 0.9317660066557425`; no success claim is supported yet.

