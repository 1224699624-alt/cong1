# R075/R076 Layout Audit Summary

Date: 2026-06-26

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## R075 Image-Evidence Layout Scorer

R075 tested whether local image evidence can safely remove bridge-like pixels from the strong R025b/R038-style masks.

Validation was done on the first 24 original-val images only. Parameters were selected on this val subset, not on clean-test-v2.

| System | Images | Dice | Precision | Recall | Boundary IoU | False Bridge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Anchor val24 | 24 | `0.883316` | `0.867348` | `0.907675` | `0.229799` | `0.083333` |
| R075 best val24 | 24 | `0.883303` | `0.867552` | `0.907435` | `0.230016` | `0.041667` |
| Delta | 24 | `-0.000013` | `+0.000204` | `-0.000240` | `+0.000218` | `-0.041667` |

Interpretation: R075 can reduce bridge-like flags, but the recall loss fully cancels the gain. The effect size is effectively zero, so a full-val and clean-test-v2 application is not justified.

Decision: mark R075 as `DONE_FLAT`. Do not continue hand-tuned bridge-removal postprocessing.

## R076 Train/Val Label-Layout Audit

R076 audited original train/val label geometry without using clean-test-v2 for selection.

Artifacts:

- `outputs/analysis/r076_trainval_label_layout_audit.json`
- `configs/dataset_filters/tsrs_rsna_epiphysis_layout_audit_filtered_v1_draft.json`
- script: `scripts/analyze_trainval_label_layout_audit.py`

Key results:

- Train labels audited: `875`
- Val labels audited: `96`
- Strong train layout outliers at threshold `6.0`: `3`
- Draft train exclusions: `6079`, `1683`, `1473`

Top train layout outliers:

| Image | Outlier Score | Main Signal |
| --- | ---: | --- |
| `6079.png` | `15.146` | very large median component area / thickness, low component count |
| `1683.png` | `7.632` | shifted vertical layout, low component count, compressed height |
| `1473.png` | `6.591` | low component count and compressed layout height |

Worst val anchor cases:

| Image | Anchor Dice | Layout Score |
| --- | ---: | ---: |
| `2246.png` | `0.657326` | `6.167` |
| `1418.png` | `0.695031` | `4.830` |
| `1884.png` | `0.766064` | `3.001` |
| `1518.png` | `0.766167` | `7.077` |
| `5882.png` | `0.775325` | `2.283` |

Interpretation: there are real layout outliers, but the train-side issue is tiny (`3/875`). This is too small to explain the remaining `+0.0174` Dice gap by itself. A filtered dataset variant excluding only these samples is a low-risk control, but it should not be expected to solve the target.

## Decision

Do not launch a heavy GPU training run solely for the R076 three-sample filter unless a visual check confirms obvious label corruption. The next high-value path should move away from postprocessing and broad filtering toward one of:

1. visually confirmed label repair / tiny isolated filter control;
2. a learned layout-quality model trained on failure-localized patches rather than deleting data;
3. a new architecture that uses stronger anatomical priors before mask decoding.
