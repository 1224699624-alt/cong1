# R320 Paper-Ready Experiment Tables

## Evaluation protocol

- Dataset: TSRS_RSNA-Epiphysis only.
- Split: 96-case original validation set.
- Evaluation: R201; clean-test-v2 was not used.
- Values: case-wise mean +/- sample standard deviation.
- Evidence boundary: current backbone results use one training seed. The SD is across cases, not across seeds.

## Main comparison

| Backbone | Method | Dice (%) | IoU (%) | B-IoU (%) | B-F1 (%) | SDice@2 (%) | SDice@5 (%) | HD95 (px) | ASSD (px) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | Baseline | 86.66 +/- 6.11 | 76.91 +/- 8.42 | 15.02 +/- 7.02 | 25.50 +/- 10.24 | 46.88 +/- 16.84 | 78.95 +/- 16.47 | 26.63 +/- 79.09 | 7.37 +/- 20.93 |
| U-Net | + Ours | 87.35 +/- 5.81 | 77.97 +/- 8.22 | 16.99 +/- 6.79 | 28.50 +/- 9.61 | 51.66 +/- 15.36 | 83.15 +/- 12.84 | 21.86 +/- 70.02 | 5.81 +/- 17.12 |
| nnU-Net v2 2D | Baseline | 89.71 +/- 6.73 | 81.92 +/- 9.63 | 23.99 +/- 8.76 | 37.90 +/- 11.39 | 65.45 +/- 16.47 | 89.65 +/- 11.81 | 54.34 +/- 211.79 | 14.17 +/- 61.70 |
| nnU-Net v2 2D | + Ours (R317) | 90.29 +/- 5.05 | 82.64 +/- 7.60 | 24.52 +/- 8.38 | 38.67 +/- 10.76 | 66.84 +/- 15.29 | 91.33 +/- 8.87 | 14.01 +/- 57.88 | 4.05 +/- 15.41 |

## Task-specific separation diagnostics

| Backbone | Method | Precision (%) | Recall (%) | Gap FP (%) | Merge rate (%) | Component-count MAE |
|---|---|---:|---:|---:|---:|---:|
| U-Net | Baseline | 79.94 +/- 8.20 | 95.37 +/- 5.03 | 44.82 +/- 15.63 | 83.33 +/- 37.46 | 4.27 +/- 3.18 |
| U-Net | + Ours | 82.13 +/- 7.25 | 93.90 +/- 6.21 | 38.36 +/- 12.92 | 66.67 +/- 47.39 | 2.72 +/- 2.60 |
| nnU-Net v2 2D | Baseline | 88.65 +/- 9.78 | 91.62 +/- 5.05 | 20.55 +/- 9.11 | 57.29 +/- 49.73 | 5.67 +/- 25.38 |
| nnU-Net v2 2D | + Ours (R317) | 90.93 +/- 6.54 | 90.18 +/- 6.19 | 18.17 +/- 9.07 | 50.00 +/- 50.26 | 1.85 +/- 2.40 |

Gap-region FP, merge rate, and component-count MAE are task-specific diagnostic metrics and should be presented separately from standard segmentation metrics.

The U-Net comparison is an exact paired run with shared initialization. The nnU-Net comparison comes from the mature R260/R317 experiment chain under the same R201 original-val protocol; it should not be described as a newly repeated multi-seed R320 pair.

## Alpha ablation

| Route | Alpha | Dice (%) | IoU (%) | HD95 (px) | ASSD (px) | Gap FP (%) | Merge rate (%) | Gate | Selected |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| continuous | 0.020 | 90.36 +/- 5.04 | 82.77 +/- 7.61 | 13.98 +/- 57.98 | 4.08 +/- 15.63 | 19.00 +/- 9.35 | 55.21 +/- 49.99 | Yes | No |
| continuous | 0.035 | 90.33 +/- 5.06 | 82.71 +/- 7.63 | 13.94 +/- 57.97 | 4.07 +/- 15.54 | 18.43 +/- 9.18 | 52.08 +/- 50.22 | Yes | Yes |
| gated | 0.020 | 90.31 +/- 5.05 | 82.68 +/- 7.60 | 14.00 +/- 57.87 | 4.15 +/- 15.40 | 18.68 +/- 9.25 | 52.08 +/- 50.22 | No | No |
| gated | 0.035 | 90.21 +/- 5.01 | 82.50 +/- 7.54 | 13.96 +/- 57.87 | 4.09 +/- 15.28 | 17.94 +/- 8.94 | 48.96 +/- 50.25 | No | No |

The continuous route with alpha=0.020 has the highest Dice/IoU, whereas alpha=0.035 has slightly lower HD95/ASSD and was selected by the predeclared distance-focused gate. These are single-seed development results.

## Reporting constraints

- Use foreground IoU, not mIoU, in the manuscript unless a separately defined class-mean IoU is computed.
- Do not report the case-wise SD as a seed-wise SD.
- R260's 0.05 coefficient is a different background-separation weight and is not part of the R316c alpha sweep.
- R317 epoch rows are checkpoint-maturity diagnostics and belong in supplementary material, not the primary alpha-ablation table.

## Artifacts

- `outputs/analysis/r320_paper_main_results.csv`
- `outputs/analysis/r320_paper_alpha_ablation.csv`
- `outputs/analysis/r320_paper_checkpoint_ablation.csv`
- `outputs/analysis/r320_paper_tables.tex`
- `scripts/export_r320_paper_tables.py`
