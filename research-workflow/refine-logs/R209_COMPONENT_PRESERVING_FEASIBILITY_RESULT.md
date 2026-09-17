# R209 Component-Preserving Feasibility Result

Date: 2026-07-06

## Question

After R205-F0 showed that gap-pixel deletion can improve overlap/boundary metrics while fragmenting anatomy, is there still a viable next route that targets bone-gap adhesion and component errors without free pixel deletion?

## Protocol

- Workflow: ARIS train/val-only feasibility audit.
- Dataset: `TSRS_RSNA-Epiphysis`.
- Split: original `val`.
- Anchor masks: `outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis/val/masks`.
- Ground truth: `data/raw/TSRS_RSNA-Epiphysis/val_labels`.
- clean-test-v2 usage: `false`.
- Script: `scripts/run_r209_component_preserving_feasibility_audit.py`.
- Outputs:
  - `outputs/analysis/r209_component_preserving_feasibility_audit_val.json`
  - `outputs/analysis/r209_component_preserving_feasibility_audit_val.csv`
  - `outputs/bridge_logs/r209_component_val_fast.log`

Implementation note: the first full run was CPU-bound because component-to-instance matching repeatedly scanned full images. The script was optimized to compute a prediction-component by GT-instance overlap matrix once per image. A val8 smoke check exactly matched the previous val8 summary before rerunning full val.

## Full Val Result

Decision: `go_component_preserving_model_feasible`

Key validation summary:

- `num_images`: `96`
- `missing`: `0`
- anchor Dice: `0.895723`
- anchor IoU: `0.814783`
- anchor Recall: `0.892619`
- anchor Boundary IoU: `0.222945`
- anchor gap-region FP rate: `0.188199`
- anchor component merge rate: `0.343750`
- anchor component count MAE: `3.020833`
- images with predicted multi-instance merge: `72/96`
- predicted merge image rate: `0.750000`
- mean merged predicted components per image: `1.156250`
- mean predicted component count: `22.781250`
- mean GT instance count: `25.562500`
- mean covered GT instances: `25.385417`

The oracle values are the binary GT upper bound and are not a deployable model result. They are used only to test whether the failure mode exists and whether component-safe instance supervision has room to improve.

## Distribution Notes

- `merged_pred_components`: median `1`, q75 `2`, max `4`.
- `anchor_component_count_mae`: median `1`, q75 `4`, max `21`.
- `anchor_gap_region_fp_rate`: median `0.192222`, q75 `0.239691`, max `0.404282`.
- `anchor_boundary_iou`: median `0.217944`, q75 `0.287937`, max `0.436696`.
- `anchor_dice`: median `0.912144`, q25 `0.897396`, max `0.949597`.

Top merge-heavy validation cases:

| Image | Merged pred comps | Count MAE | Gap FP | Dice | Boundary IoU |
|---|---:|---:|---:|---:|---:|
| `3700.png` | 4 | 5 | 0.228686 | 0.878876 | 0.164120 |
| `3460.png` | 4 | 3 | 0.238711 | 0.933848 | 0.207283 |
| `1886.png` | 3 | 20 | 0.261819 | 0.903225 | 0.117597 |
| `1931.png` | 3 | 8 | 0.314112 | 0.917213 | 0.131782 |
| `1742.png` | 3 | 7 | 0.318610 | 0.900461 | 0.105695 |
| `3725.png` | 3 | 3 | 0.078392 | 0.927540 | 0.340778 |
| `1880.png` | 3 | 1 | 0.155963 | 0.912144 | 0.198632 |
| `1901.png` | 3 | 0 | 0.302551 | 0.915544 | 0.135766 |

## Interpretation

R209 supports a component-preserving instance-level route. The relevant evidence is not that the GT oracle is perfect, but that R110 train/val masks frequently merge multiple GT instances into one predicted connected component while still covering most GT instances. That is the exact failure pattern needed for an instance-separation model or component-preserving decoder/gate.

R209 does not support another broad `anatomy-aware` or gap-pixel deletion run. R205-F0 already showed that deleting gap-region false positives can create attractive metric gains while badly fragmenting components. R209 says the next model must preserve component/instance identity as a first-class constraint.

## R210 Direction

Proceed to R210 only if it follows these constraints:

- Train and select on train/val only.
- Use true R110 train/val masks as the anchor input.
- Predict instance/component separation or component-safe edits, not free pixel deletion.
- Include a validation gate requiring:
  - nonzero edits on merge-heavy val cases,
  - Dice/IoU/Recall not materially below the R110 anchor,
  - reduced gap FP or component merge,
  - no increase in component count MAE,
  - hard-case visualization showing no over-erosion or missing epiphyses.
- Apply clean-test-v2 once only after the validation gate is locked.

Recommended R210 candidates:

1. Component-preserving instance-separation head: train a small head on image, R110 mask, distance/boundary/gap maps, and GT instance labels to predict split likelihood inside merged components.
2. Component-safe acceptance gate: accept a split only if it keeps or improves predicted component count relative to expected GT-instance priors on val.
3. Hard-case validation panel from the R209 top merge cases before any clean-test-v2 evaluation.

Decision: move from R209-F0 to R210 design. Do not relaunch R205-style anatomy-aware pixel deletion or morphology sweeps.
