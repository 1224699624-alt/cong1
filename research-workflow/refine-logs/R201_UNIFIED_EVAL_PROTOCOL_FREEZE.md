# R201 Unified Evaluation Protocol Freeze

## Purpose

Freeze the paper-facing evaluation protocol for `TSRS_RSNA-Epiphysis_clean_test_v2/test`.
The target claim is not generic Dice chasing: the method must preserve Dice/IoU while improving boundary quality and bone-gap/component separation diagnostics.

## Frozen Data

- Dataset: `TSRS_RSNA-Epiphysis_clean_test_v2`
- Split: `test`
- Ground truth: `data/raw_variants/TSRS_RSNA-Epiphysis_clean_test_v2/test_labels`
- Expected images: `81`
- Rule: clean-test-v2 is final evaluation/diagnostic only; no threshold search, model selection, or tuning is allowed on this split.

## Metric Layers

- Main table standard metrics: Dice, IoU/Jaccard, Precision, Recall.
- Boundary metrics: Boundary IoU, Boundary F1, Surface Dice at `2 px`, Surface Dice at `5 px`, HD95 in pixels, ASSD in pixels.
- Task-specific diagnostics: gap-region FP rate, component merge rate, component count MAE.

## Implementation Details

- Binary mask rule: any pixel value `> 0` is foreground.
- Resize rule: prediction masks with mismatched shape are resized to GT shape with nearest-neighbor interpolation.
- Boundary IoU and Boundary F1 use the existing project-compatible XOR boundary with kernel `3`.
- Gap region is `dilation(GT, 9x9) - GT`.
- Surface metrics use pixel spacing because reliable physical spacing is unavailable.
- Empty-surface convention: if both prediction and GT are empty, Surface Dice/HD95/ASSD are perfect; if only one is empty, Surface Dice is `0` and HD95/ASSD are `null`.

## Metric Direction

```json
{
  "dice": "higher",
  "iou": "higher",
  "precision": "higher",
  "recall": "higher",
  "specificity": "higher",
  "boundary_iou": "higher",
  "boundary_f1": "higher",
  "surface_dice_2px": "higher",
  "surface_dice_5px": "higher",
  "hd95_px": "lower",
  "assd_px": "lower",
  "gap_region_fp_rate": "lower",
  "gap_region_precision": "higher",
  "component_merge_rate": "lower",
  "component_count_mae": "lower",
  "component_delta_mean": "closer_to_zero"
}
```

## Regression Anchor

R200 R110-vs-ARAA is the fixed regression anchor. The R201 script must reproduce R110/ARAA Dice, IoU, and Boundary IoU before its comparisons are trusted.

- Regression passed: `True`
- Unified table: `outputs\analysis\r201_unified_eval\r201_unified_comparison_table.csv`
- Unified JSON: `outputs\analysis\r201_unified_eval\r201_unified_comparison_table.json`
