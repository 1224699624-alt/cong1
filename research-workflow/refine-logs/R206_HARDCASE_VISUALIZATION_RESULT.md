# R206 Hard-Case Visualization Result

Date: 2026-07-06

## Purpose

Generate qualitative hard-case panels for the current anatomy-consistency goal: inspect bone-gap adhesion, boundary error, component merge, fragmentation risk, and recall risk without using the panels for threshold tuning.

This is diagnostic only. It does not modify predictions and does not tune on clean-test-v2.

## Inputs

- Failure manifest: `outputs/analysis/r203_failure_manifest/r203_failure_manifest.csv`
- Images: `data/raw/TSRS_RSNA-Epiphysis/test`
- GT: `data/raw_variants/TSRS_RSNA-Epiphysis_clean_test_v2/test_labels`
- Local masks available:
  - ARAA: `outputs/ablations_variants/araa_danet_epoch98/.../masks`
  - R110: `outputs/ablations_variants/r110_r100_r108_patch_basic/.../masks`
  - R202 nnU-Net: `outputs/ablations_variants/r202_nnunet2d/.../masks`

## Outputs

- Script: `scripts/build_r206_hardcase_visualizations.py`
- HTML browser index: `outputs/analysis/r206_hardcase_visualizations/index.html`
- JSON index: `outputs/analysis/r206_hardcase_visualizations/r206_hardcase_visualization_index.json`
- CSV index: `outputs/analysis/r206_hardcase_visualizations/r206_hardcase_visualization_index.csv`
- Panels: `outputs/analysis/r206_hardcase_visualizations/panels/`

## Case Counts

Generated `32` panels:

- `clean_bridge_candidate`: 8
- `fragmentation_risk_high_component_error`: 8
- `mixed_or_low_priority`: 8
- `gap_fp_candidate`: 6
- `recall_risk_do_not_cut_aggressively`: 2

## Visualization Legend

- GT boundary: green
- Prediction boundary: magenta
- True positive: blue tint
- False positive: red tint
- False negative: yellow tint

## Baselines Included

The current regenerated package includes all three available clean-test-v2 mask sets:

- ARAA
- R110
- R202 nnU-Net risk-audit baseline

Missing model count is now `0`.

## Next Use

Use the panels to manually inspect whether candidate improvements would:

- reduce visible bone-gap adhesion,
- preserve epiphysis completeness,
- avoid excessive erosion,
- avoid component fragmentation,
- improve boundary alignment.

Do not use this visualization package for clean-test-v2 model selection.
