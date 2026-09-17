# R311 clean-panel standard vs prior nnU-Net paired comparison

Date: 2026-07-27

## Purpose

Directly compare the mature clean-panel standard nnU-Net (R310) with the mature clean-panel prior nnU-Net (R308) without rerunning an already completed configuration.

## Paired protocol

- Dataset: `TSRS_RSNA-Epiphysis` only.
- Clean selection: identical 814 train / 94 original-val cases.
- Excluded validation cases: `2246`, `5882`.
- Output target: binary bone/background in both arms.
- Standard arm: one-channel X-ray, no prior input and no prior loss.
- Prior arm: X-ray plus frozen R259 relation-prior channel, prior loss weight `0.05`.
- Both arms use the same nnU-Net 2.8.1 2D backbone family, patch `1536 x 1024`, batch size 1, seed 260 and best-checkpoint inference.
- Each arm was initialized from its corresponding mature pre-clean checkpoint: R202 standard or R260 prior.
- Evaluation: identical 94-case R201 original-val clean subset.
- Test and clean-test-v2: not used.

## R201 result

| Metric | R310 clean standard nnU-Net | R308 clean prior nnU-Net | Prior - standard | Preferred |
|---|---:|---:|---:|---|
| Dice | 0.907059 | 0.906858 | -0.000201 | standard |
| IoU | 0.832534 | 0.832050 | -0.000483 | standard |
| Precision | 0.911726 | 0.917879 | +0.006153 | prior |
| Recall | 0.906142 | 0.899361 | -0.006782 | standard |
| Boundary IoU | 0.247588 | 0.246876 | -0.000712 | standard |
| Boundary F1 | 0.389773 | 0.388826 | -0.000948 | standard |
| gap-region FP rate | 0.186640 | 0.173044 | -0.013597 | prior |
| component merge rate | 0.595745 | 0.521277 | -0.074468 | prior |
| component count MAE | 1.840426 | 1.734043 | -0.106383 | prior |
| Surface Dice 2 px | 0.671691 | 0.671301 | -0.000390 | standard |
| Surface Dice 5 px | 0.914824 | 0.917256 | +0.002431 | prior |
| HD95 px | 10.281937 | 7.827458 | -2.454479 | prior |
| ASSD px | 2.955085 | 2.571234 | -0.383851 | prior |

## Per-case behavior

- The prior lowers gap-region FP on all 94 validation cases relative to the clean standard model.
- The prior improves component merge state on eight cases, worsens it on one, and leaves 85 unchanged.
- Precision improves on 93/94 cases, while recall decreases on 92/94 cases.
- Dice improves on 36/94 and decreases on 58/94 cases.
- The largest prior benefit is `1518.png`: Dice `0.804336 -> 0.886784`, HD95 `257.103578 -> 6.744490` px, and ASSD `41.482740 -> 14.744367` px.

## Interpretation

The prior has a real separation and large-error-control effect: it reduces foreground leakage into narrow gaps, lowers merge rate and improves coarse surface agreement. However, the current prior/loss is over-conservative and suppresses true bone pixels as well, causing a recall loss of `0.006782`. It therefore does not yet satisfy the desired condition that Dice/Recall remain non-inferior while separation improves.

This comparison does not prove a causal gain from the prior channel alone because the two arms inherit their corresponding mature one-channel and two-channel checkpoints. It is nevertheless the correct end-to-end comparison of the mature standard and mature prior systems on the same cleaned data and evaluation set.

## Artifacts

- Standard R201: `outputs/analysis/r310_clean_finetune_plain_nnunet_val94_r201.json`
- Prior R201: `outputs/analysis/r308_clean_finetune_clean_val94_r201.json`
- Twelve fixed hard-case panels: `outputs/visualizations/r311_clean_plain_vs_prior_nnunet_val94/`

