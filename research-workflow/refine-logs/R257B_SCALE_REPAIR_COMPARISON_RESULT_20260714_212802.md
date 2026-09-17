# R257b Scale Repair Comparison Result

## Status

- Scope: `TSRS_RSNA-Epiphysis_contrast_v1` train/original-val only
- Frozen center source: R257 fixed epoch-8 checkpoint
- Compared: instance-balanced log-scale head vs prediction-derived foreground basin scale
- clean-test-v2 used: no
- Articular-Surface used: no
- Decision: `allow_r258_with_both_scale_methods`

All frozen-protocol checks passed. Twelve shared center/proposal metrics were exactly identical to R257 (`delta <= 1e-12`), so the comparison changes scale estimation only.

## Shared image-only center results

- Center recall: `0.983693`
- Center precision: `0.921696`
- Normalized localization error: `0.114630`
- Close-pair coverage: `0.953725`
- Contact-pair coverage: `0.906250`
- Count MAE: `1.968750`
- False proposals/image: `2.135417`

## Scale comparison

| Method | Global median abs log error | Close endpoint | Contact endpoint | Gate |
|---|---:|---:|---:|---|
| R257 sigmoid scale reference | 1.451902 | 2.161196 | 2.220379 | fail |
| Instance-balanced log-scale | 0.285411 | 0.320659 | 0.458877 | pass |
| Prediction-derived basin scale | 0.117941 | 0.096814 | 0.101343 | pass |

Both repairs satisfy the preregistered `0.50/0.60/0.60` gates. Basin scale is substantially better and requires no learned scale head: it assigns frozen predicted foreground pixels to their nearest frozen predicted center and uses square-root basin area. The learned log-scale head remains a valid independent ablation/backup.

## Artifact integrity

- Frozen R257 checkpoint SHA256: `ccb168f40b5315e0e7523b622482a309255d2a709dee0e1d009c2958264d6717`
- Log-scale checkpoint: `outputs/scale_repair/r257b_scale_repair_fullval/log_scale_head_seed2571_20260714_211948.pt`
- Log-scale checkpoint SHA256: `aa6c6eb2a24e3da1285dee51b8b26041798d3cdea14106627c43df576cc9d3f9`

## Decision

R258 exploratory OOF proposal generation is allowed. Use basin scale as the primary prediction-derived scale because it is more accurate and contains no additional learned head; retain log-scale as a preregistered ablation. R258 must generate patient-disjoint OOF train proposals and image-only original-val proposals before retraining the simplified age/sex-conditioned relation prior. This result does not authorize nnU-Net integration or clean-test-v2 use.
