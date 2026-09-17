# R310 clean standard nnU-Net status

Date: 2026-07-25

- Dataset: TSRS_RSNA-Epiphysis only.
- Clean keep-list: 814 train / 94 val.
- Model: standard one-channel nnU-Net 2.8.1.
- Input: X-ray only.
- Relation prior channel: none.
- Prior loss: none.
- Instance auxiliary loss: none.
- Initialization: mature R202 standard nnU-Net checkpoint.
- Checkpoint SHA256: `65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7`.
- Strict network-key loading: passed.
- Patch: 1536 x 1024; batch size 1 for local 6 GiB GPU.
- Frozen baseline inference: completed on all 94 clean-val cases.
- Clean fine-tuning: completed.
- Best EMA pseudo Dice: `0.9219656123` at completed epoch 5.
- Early stopping: stopped after 15 completed epochs and 10 epochs without a new best.
- Final evaluation used `checkpoint_best.pth`, not the last epoch.
- Original-val clean subset evaluation: completed on all 94 retained cases.
- Comparison visualizations: 12 cases rendered.
- Test/clean-test-v2: not used.

Outputs:

- `outputs/analysis/r310_frozen_plain_nnunet_clean_val94_r201.json`
- `outputs/analysis/r310_clean_finetune_plain_nnunet_val94_r201.json`
- `outputs/visualizations/r310_plain_nnunet_frozen_vs_clean_val94/`

## R201 results

| Metric | Frozen standard nnU-Net | Clean fine-tuned standard nnU-Net | Delta |
|---|---:|---:|---:|
| Dice | 0.907527 | 0.907059 | -0.000468 |
| IoU | 0.833216 | 0.832534 | -0.000682 |
| Precision | 0.903362 | 0.911726 | +0.008365 |
| Recall | 0.915288 | 0.906142 | -0.009146 |
| Boundary IoU | 0.245366 | 0.247588 | +0.002223 |
| Boundary F1 | 0.386999 | 0.389773 | +0.002774 |
| gap-region FP rate | 0.212666 | 0.186640 | -0.026025 |
| component merge rate | 0.648936 | 0.595745 | -0.053191 |
| component count MAE | 2.021277 | 1.840426 | -0.180851 |
| Surface Dice 2 px | 0.667456 | 0.671691 | +0.004235 |
| Surface Dice 5 px | 0.913694 | 0.914824 | +0.001130 |
| HD95 px | 10.940608 | 10.281937 | -0.658671 |
| ASSD px | 2.734615 | 2.955085 | +0.220471 |

## Interpretation

- Clean fine-tuning shifted the standard nnU-Net toward a more conservative foreground prediction: precision increased, but recall decreased on 92/94 cases.
- Gap FP decreased on all 94 cases; merge rate improved in 5 cases and was unchanged in 89 cases.
- Local boundary metrics improved modestly, but Dice was non-inferior only in an approximate descriptive sense and decreased by `0.000468` overall.
- The mean HD95 improvement is partly driven by a few hard cases, especially `1884.png`; it is not a uniform improvement.
- `1518.png` became a major failure after standard clean fine-tuning: Dice fell from `0.885671` to `0.804336`, and ASSD increased from `16.342898` to `41.482740`.

## Comparison with R308 clean prior nnU-Net

R308 uses the same retained 94 validation cases but includes the R259 prior input and prior loss. Relative to R310 clean standard nnU-Net:

- Dice: `0.906858` vs `0.907059` (`-0.000201`).
- Recall: `0.899361` vs `0.906142` (`-0.006782`).
- Boundary IoU: `0.246876` vs `0.247588` (`-0.000712`).
- gap-region FP rate: `0.173044` vs `0.186640` (`-0.013597`, better).
- component merge rate: `0.521277` vs `0.595745` (`-0.074468`, better).
- Surface Dice 5 px: `0.917256` vs `0.914824` (`+0.002431`, better).
- HD95: `7.827458` vs `10.281937` (`-2.454479` px, better).
- ASSD: `2.571234` vs `2.955085` (`-0.383851` px, better).

The prior version therefore has a clearer separation/topology and large-error-control advantage, but it obtains this by further suppressing foreground and losing recall. It is not a complete win because Dice and the 2 px boundary metrics do not improve over the clean standard model.
