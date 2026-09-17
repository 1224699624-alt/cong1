# R308 clean-panel R260 experiment status

Date: 2026-07-24

## Cleaned-data interpretation

- Review source: `outputs/manual_review/tsrs_epiphysis_label_cleaning_v1/panels`.
- The panel JPEGs are review composites and are not model inputs.
- Remaining panel stems are treated as the manual keep-list.
- Kept: 814/875 train, 94/96 original-val.
- Excluded: 61 train and two val cases (`2246`, `5882`).
- The copied `working_labels` contain no detected pixel edits relative to available originals; cleaning is therefore a case-selection experiment rather than a relabeling experiment.
- Test panels and test labels are not used.

## R308 design

- Baseline: frozen, recovered R260 best.
- Experimental arm: the same R260 best continued on 814 clean-train cases.
- Validation/model selection: fixed 94 clean-val cases.
- Output remains binary bone/background.
- R260 frozen relation prior and prior loss weight `0.05` are unchanged.
- No color-instance auxiliary loss is used in this first cleaning ablation.
- R201 compares frozen R260 and clean-finetuned R260 on the identical 94-case clean-val subset.

## Provenance

- R260 checkpoint SHA256: `7d1d8d2916115c2e29696b2a76881da2a7a78f3db91a5f6a9491297369052a6e`.
- Reconstructed R259 priors: 875 train and 96 val; all nonzero and provenance hashes validated.
- nnU-Net: 2.8.1.
- GPU: local NVIDIA RTX 3060 Laptop GPU, 6 GiB.
- Patch: exact R260 `1536 x 1024`.
- Architecture: exact R260 9-stage PlainConvUNet.
- Local batch size: 1 instead of the original plan's 2, solely for 6 GiB VRAM compatibility.
- Data augmentation workers: 0 (single-threaded), required for Windows handle stability.

## Final state

- Dataset integrity: passed.
- Preprocessing: completed for all 908 retained train+val cases.
- Strict R260 checkpoint loading: passed; all keys loaded and prior channel nonzero.
- Training stopped at epoch 51.
- Best EMA pseudo Dice: `0.919876895`.
- Inference, R201 evaluation, and 12-panel visualization completed.
- No OOM or NaN after the single-threaded Windows restart.
- `clean-test-v2`: not used.

## Clean-val94 R201 result

| Metric | Frozen R260 | Clean fine-tune | Delta |
|---|---:|---:|---:|
| Dice | 0.906476 | 0.906858 | +0.000382 |
| Recall | 0.898986 | 0.899361 | +0.000375 |
| Boundary IoU | 0.246645 | 0.246876 | +0.000231 |
| Boundary F1 | 0.388607 | 0.388826 | +0.000219 |
| Gap-region FP rate | 0.173229 | 0.173044 | -0.000186 |
| Component merge rate | 0.510638 | 0.521277 | +0.010638 (worse) |
| Surface Dice 2 px | 0.671212 | 0.671301 | +0.000089 |
| Surface Dice 5 px | 0.916639 | 0.917256 | +0.000617 |
| HD95 px | 10.052914 | 7.827458 | -2.225456 |
| ASSD px | 2.697873 | 2.571234 | -0.126639 |

The HD95/ASSD gains are dominated by recovery of the extreme `1518.png` failure. Dice improved on 48/94 images and worsened on 46/94; component merge improved on one case and worsened on two. The clean selection is mildly beneficial but is not a complete solution to adjacency errors.

## Expected outputs

- `outputs/analysis/r308_frozen_r260_clean_val94_r201.json`
- `outputs/analysis/r308_clean_finetune_clean_val94_r201.json`
- `outputs/visualizations/r308_frozen_vs_clean_finetune_val94/`
- `outputs/bridge_logs/r308_clean_panels_r260_local.log`
