# R161 Reannotated Train/Val-Only Variant

## Purpose

After R159/R160 closed the current architecture/environment routes, R161 checked whether the existing reannotated dataset can safely support a data branch without violating the success-evaluation rule.

Success evaluation remains only `TSRS_RSNA-Epiphysis_clean_test_v2/test`.

## Audit

Source reannotated variant:

`data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1`

Counts:

- train: `1256` images / `1256` labels
- val: `96` images / `96` labels
- test: `122` images / `122` labels

Overlap with original dataset:

- reannotated train overlaps original train: `875`
- reannotated train extras vs original: `381`
- reannotated val overlaps original val: `96`
- reannotated test overlaps original test: `97`
- reannotated test extras vs original: `25`

Decision from audit: reannotated test is unsafe for success claims and must be excluded from any training/evaluation launcher used for target claims.

Audit JSON:

`outputs/analysis/r161_reannotated_variant_audit.json`

## Built Variant

Created isolated variant:

`data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1`

Policy:

- train: copied from reannotated variant
- val: copied from reannotated variant
- test: copied from original `data/raw/TSRS_RSNA-Epiphysis`
- reannotated test: excluded
- success evaluation: must remain clean-test-v2/test

Created counts:

- train: `1256` images / `1256` labels
- val: `96` images / `96` labels
- test: `97` images / `97` labels

Verification:

- target test equals original test: `true`
- target test equals reannotated test: `false`
- target test extra vs original: `0`

Build metadata:

`data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1/reannotated_trainval_only_metadata.json`

Build JSON:

`outputs/analysis/r161_reannotated_trainval_only_variant_build.json`

## Decision

R161 opens a safe data branch, but it does not prove model quality.

Next GPU run, if chosen, must:

- train only on `TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1` train/val;
- never use `TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1/test`;
- evaluate success only on `TSRS_RSNA-Epiphysis_clean_test_v2/test`;
- keep original test only as a secondary control.

