# R065 Context Segmenter Result Summary

**Date**: 2026-06-26

## Run

- Run ID: `R065`
- Method: compact context segmenter, direct image-to-mask prediction
- Target evaluation: `TSRS_RSNA-Epiphysis_clean_test_v2/test`
- Control evaluation: original `TSRS_RSNA-Epiphysis/test`
- Remote output:
  - `outputs/analysis/r065_context_segmenter_clean_test_v2_metrics.json`
  - `outputs/analysis/r065_context_segmenter_original_test_metrics.json`
  - `outputs/bridge_logs/r065_context_segmenter_gpu0.log`

## Primary Result

| System | clean-test-v2 Dice | Delta vs R038 | Delta vs target | Verdict |
| --- | ---: | ---: | ---: | --- |
| R038 current best | `0.914357` | - | `-0.017409` | baseline |
| R065 compact context segmenter | `0.889453` | `-0.024904` | `-0.042313` | negative |
| Target | `>0.9317660066557425` | `+0.017409` over R038 | - | unmet |

Secondary clean-test-v2 metrics:

- IoU: `0.801692`
- Precision: `0.836930`
- Recall: `0.951533`
- Boundary IoU: `0.175126`
- Component-count error: `2.753086`
- False-bridge flag rate: `0.728395`

Original-test control:

- Dice: `0.874035`
- Precision: `0.820827`
- Recall: `0.937442`
- Boundary IoU: `0.167902`

## Result-To-Claim Verdict

- Claim tested: C2, "long-range context architecture adds missing image evidence."
- Local verdict: **not supported**.
- Reason: R065 is a genuinely new direct architecture route, but it underperforms R038 by `0.024904` Dice on the required clean-test-v2 split and misses the target by `0.042313`.
- Integrity note: success evaluation used `clean-test-v2`; the new reannotated test split was not used for the claim.

## Gate Decision

R066 is **not launched** under the written gate because R065 did not beat R038 (`0.889453 <= 0.914357`).

R067 should **not be auto-launched** from the current plan. The fallback was intended for a partially promising R065/R066 branch; R065 is a clear negative result, not a near-miss.

## Interpretation

R065 shows the direct segmenter can learn a high-recall mask, but it does not recover the precision and boundary quality needed for the target. The high false-bridge flag rate (`0.728395`) and low boundary IoU (`0.175126`) suggest that the compact context model blurs separated epiphysis structures rather than resolving them.

This result weakens the current "compact context segmenter" architecture hypothesis. Further automatic loss ablations on the same model are unlikely to close a `0.042313` Dice gap to target.

## Recommended Next ARIS Move

Do not spend the next GPU run on R066 as originally gated. The next plan should be revised before launching more training:

1. Use the R065 per-image failure profile to identify whether the dominant loss is bridges, missed small components, or global overmasking.
2. Search for a stronger architecture direction with explicit instance/separation modeling, not only long-range context.
3. Consider returning to data/label audit if literature-driven architecture ideas continue to lose to R038.

