# R077 Layout-Filtered Control Summary

Date: 2026-06-26

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Setup

R076 found three strong train-side layout outliers: `6079`, `1683`, and `1473`. Visual inspection confirmed that these are not normal epiphysis labels:

- `6079`: wrist/carpal region is labeled instead of the expected epiphysis layout.
- `1683`: extreme crop/low-quality image with only tiny partial labels.
- `1473`: very low-contrast image with sparse partial labels.

R077 created an isolated dataset variant:

- Dataset: `TSRS_RSNA-Epiphysis_layout_audit_filtered_v1`
- Train: `872` kept / `3` removed
- Val: unchanged (`96`)
- Test: unchanged (`97`)

The original dataset was not modified.

R077 then trained the R068 HoVer-style instance-separation segmenter on this filtered variant and evaluated:

- target: `TSRS_RSNA-Epiphysis_clean_test_v2/test`
- control: `TSRS_RSNA-Epiphysis_layout_audit_filtered_v1/test`

## Results

| Run | clean-test-v2 Dice | Precision | Recall | Boundary IoU | False Bridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| R068 original instance-separation | `0.903030` | `0.879287` | `0.929789` | `0.203925` | `0.580247` |
| R077 layout-filtered control | `0.896007` | `0.866136` | `0.929998` | `0.190438` | `0.728395` |

Control original-test Dice for R077: `0.880698`.

Best validation epoch was `10` with val Dice `0.876628`, below the R068 validation peak (`0.885080`).

## Interpretation

The three excluded labels are real data-quality problems, but removing them is not enough to improve generalization. R077 reduces neither the target gap nor the bridge profile; it is worse than R068 and far below R038/R070.

This closes tiny train-label filtering as a primary route. Data cleaning remains useful for dataset hygiene and reporting, but the next target-seeking experiment should not be another deletion/filtering run.

## Decision

Stop:

- broad data deletion;
- tiny layout-filter-only retraining;
- more R038 bridge-removal postprocessing.

Next ARIS step should pivot to a genuinely new architecture with stronger anatomical priors or structured decoding, supported by literature rather than another same-family selector/filter.
