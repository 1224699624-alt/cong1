# R164 Filtered Reannotated Variant

## Status

DONE_VARIANT_READY.

R164 builds a filtered version of the safe reannotated train/val-only dataset after R162 failed and R163 identified train-side label/protocol shift.

Variant:

`data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_filtered_trainval_v1`

## Filtering Rule

Source:

`data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1`

Rejected train labels if any of the following held:

- paired original-vs-reannotated label Dice `< 0.65`
- foreground-area ratio `< 0.45` or `> 2.25`
- absolute component-count delta `> 12`
- foreground fraction `< 0.002` or `> 0.09`
- component count `< 8` or `> 35`

Extra reannotated train labels without original pairing used only the basic foreground/component gates.

## Build Result

Selected:

- train: `1159`
- val: `95`
- test: `97`

Rejected:

- train: `97`
- val: `1`

Main train rejection reasons:

- paired Dice too low: `73`
- component count too low: `35`
- component delta too large: `19`
- foreground ratio too high: `18`
- foreground fraction too low: `16`
- foreground ratio too low: `9`
- foreground fraction too high: `6`
- component count too high: `1`

## Safety Check

Checker:

`scripts/check_r164_filtered_reannotated_variant.py`

Result:

- `ok=true`
- train images/labels/paired: `1159/1159/1159`
- val images/labels/paired: `95/95/95`
- test images/labels/paired: `97/97/97`
- variant test equals original test: `true`
- variant test equals reannotated test: `false`
- reannotated test extras vs original: `25`

## Evidence

- Dry run: `outputs/analysis/r164_filtered_reannotated_variant_dryrun.json`
- Build summary: `outputs/analysis/r164_filtered_reannotated_variant_build.json`
- Check summary: `outputs/analysis/r164_filtered_reannotated_variant_check.json`
- Metadata: `data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_filtered_trainval_v1/r164_filtered_reannotated_metadata.json`

## Decision

R164 passes the data gate. A GPU follow-up is now technically valid, but it must use a corrected launcher that:

- trains on `TSRS_RSNA-Epiphysis_reannotated_filtered_trainval_v1`;
- never uses `TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1/test`;
- evaluates success only on `TSRS_RSNA-Epiphysis_clean_test_v2/test`;
- fixes the R162 control-root issue so original-test control is non-empty, or explicitly omits that control.

Suggested next run:

`R165_filtered_reannotated_trainval_instance_sep`
