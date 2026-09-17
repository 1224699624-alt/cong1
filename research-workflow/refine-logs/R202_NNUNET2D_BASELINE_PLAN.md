# R202 nnU-Net 2D Baseline Plan

## Purpose

Run nnU-Net as the first modern, authoritative medical segmentation baseline under the frozen R201 protocol.
R202 is a baseline, not a new proposed method. It tests whether a strong self-configuring medical segmentation recipe changes the paper evidence relative to R110 and ARAA.

## Data Discipline

- Task: `TSRS_RSNA-Epiphysis` only.
- Training/model selection data: original `data/raw/TSRS_RSNA-Epiphysis/train` and `val`.
- Final inference data: clean-test-v2 image IDs from `data/raw_variants/TSRS_RSNA-Epiphysis_clean_test_v2/test_labels`, with images read from original `data/raw/TSRS_RSNA-Epiphysis/test`.
- clean-test-v2 labels are not copied into nnU-Net training labels.
- Output is isolated under `outputs/nnunet/r202` and `outputs/ablations_variants/r202_nnunet2d`.

## Implementation

- Data conversion script: `scripts/prepare_r202_nnunet_dataset.py`
- Prediction conversion script: `scripts/convert_r202_nnunet_predictions.py`
- Remote launcher: `outputs/bridge_logs/run_r202_nnunet2d_baseline.sh`
- Unified evaluator: `scripts/run_r201_unified_eval.py`

The nnU-Net dataset is prepared as:

- `outputs/nnunet/r202/nnUNet_raw/Dataset202_TSRS_RSNAEpiphysis2D/imagesTr`
- `outputs/nnunet/r202/nnUNet_raw/Dataset202_TSRS_RSNAEpiphysis2D/labelsTr`
- `outputs/nnunet/r202/nnUNet_raw/Dataset202_TSRS_RSNAEpiphysis2D/imagesTs`

The original train/val split is preserved by writing `splits_final_r202.json` and copying it to the nnU-Net preprocessed dataset as `splits_final.json` after planning/preprocessing.

## Remote Command

Run on the server from `/home/shenzeyu/workspace/YOLO_SAM_generic_src`:

```bash
screen -dmS r202_nnunet2d_gpu0 bash outputs/bridge_logs/run_r202_nnunet2d_baseline.sh
```

Before launch, check GPU availability:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

Use a free GPU or edit `CUDA_VISIBLE_DEVICES` externally before launching if needed.

## Outputs

- nnU-Net raw/preprocessed/results: `outputs/nnunet/r202`
- clean-test-v2 masks for R201: `outputs/ablations_variants/r202_nnunet2d/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks`
- base clean-test metrics: `outputs/analysis/r202_nnunet2d_clean_test_v2_metrics.json`
- R201 unified metrics: `outputs/analysis/r201_unified_eval/r202_nnunet2d_unified_metrics.json`
- refreshed R201 comparison table: `outputs/analysis/r201_unified_eval/r201_unified_comparison_table.csv`

## Success Criteria

R202 is competitive if:

- Dice and IoU are at least above ARAA, and ideally near or above R110.
- Boundary IoU/F1, Surface Dice, HD95, and ASSD are competitive with or better than R110.
- Component merge rate, component count MAE, and gap-region FP rate are better than both R110 and ARAA.

If nnU-Net improves standard metrics but not anatomy diagnostics, it is a strong baseline but does not solve the paper's target failure mode.
If nnU-Net improves anatomy diagnostics while losing too much Dice/IoU, it becomes useful as a diagnostic/complement branch rather than a main method.

## Reviewer Framing

nnU-Net should be reported as a strong medical image segmentation baseline, not as part of the proposed method.
All R202 results must be reported through the R201 protocol so that Dice/IoU, boundary metrics, and anatomical diagnostics remain comparable to R110 and ARAA.
