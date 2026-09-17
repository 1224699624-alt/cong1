# R168 Reannotation Protocol Review Package

This package is train/val only. It excludes clean-test-v2 and excludes the reannotated test split.

## Summary

- Total review rows: `96`
- Train rows: `80`
- Val rows: `16`
- Output HTML: `outputs\analysis\r168_reannotation_protocol_review_package\R168_REANNOTATION_PROTOCOL_REVIEW.html`
- Worklist CSV: `outputs\analysis\r168_reannotation_protocol_review_package\r168_reannotation_protocol_review_worklist.csv`

## Decision Values

- `original_label_correct`: 原标签更可信
- `reannotated_label_correct`: 新标注更可信
- `both_wrong_needs_correction`: 两者都不对，需要修正PNG
- `exclude_from_training_variant`: 排除，不放入训练变体
- `uncertain_second_review`: 不确定，需要二审

## Review Notes

- Original label overlay is blue with low opacity.
- Reannotated label overlay is red with low opacity.
- Difference panel uses green for overlap, blue for original-only, red for reannotated-only.
- Exported reviewed CSV should be used to decide whether a corrected isolated train/val variant is worth building.

## Preview Reviewed CSV

After exporting `r168_reannotation_protocol_review_reviewed.csv` from the HTML page, run:

```powershell
python scripts\preview_r168_reannotation_protocol_decisions.py --decisions-csv outputs\analysis\r168_reannotation_protocol_review_package\r168_reannotation_protocol_review_reviewed.csv
```

The preview is read-only. It requires at least `40` reviewed train rows, `20` usable train decisions, and `8` reviewed val rows before a later isolated-variant step is allowed.

## Build Isolated Variant After Gate Pass

The easiest path is the R172 wrapper:

```powershell
python scripts\run_r172_reannotation_protocol_pipeline.py --decisions-csv outputs\analysis\r168_reannotation_protocol_review_package\r168_reannotation_protocol_review_reviewed.csv
```

If the wrapper returns `dry_run_ok`, create the variant intentionally:

```powershell
python scripts\run_r172_reannotation_protocol_pipeline.py --decisions-csv outputs\analysis\r168_reannotation_protocol_review_package\r168_reannotation_protocol_review_reviewed.csv --create-variant --overwrite
```

If corrected PNGs are needed, add `--correction-dir <DIR>`.

First run a dry-run:

```powershell
python scripts\build_r170_reannotation_protocol_variant.py --decisions-csv outputs\analysis\r168_reannotation_protocol_review_package\r168_reannotation_protocol_review_reviewed.csv --output-json outputs\analysis\r170_reannotation_protocol_variant_build_status.json
```

Only if the dry-run status is `dry_run_ok`, create the isolated variant intentionally:

```powershell
python scripts\build_r170_reannotation_protocol_variant.py --decisions-csv outputs\analysis\r168_reannotation_protocol_review_package\r168_reannotation_protocol_review_reviewed.csv --create --overwrite --output-json outputs\analysis\r170_reannotation_protocol_variant_build_status.json
```

If any row is marked `both_wrong_needs_correction`, provide corrected label PNGs with `--correction-dir <DIR>` before creating the variant.
