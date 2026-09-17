# R210-F1 Neck Candidate Gate Result

Date: 2026-07-06

## Purpose

Continue R210 after R210-F0 rejected the cheap pixel-MLP candidate generator. R210-F1 tests whether deterministic, spatially coherent neck candidates inside R110 anchor components can reduce gap adhesion and component errors under the same component-safe validation gate.

## Implemented Script

- `scripts/run_r210_f1_neck_candidate_gate.py`

Properties:

- Dataset: `TSRS_RSNA-Epiphysis`.
- Split used: original `val`.
- Anchor: `r110_r100_r108_patch_basic_trainval`.
- clean-test-v2 used: `false`.
- Output isolation:
  - smoke: `r210_f1_neck_candidate_gate_smoke`
  - val32: `r210_f1_neck_candidate_gate_val32`
- Gate: R201 metrics + component-safe acceptance.
- Candidate generator: deterministic neck-like regions from R110 connected components, distance-transform low-percentile regions, image-gradient support, and thin/low-fill cut-shape filtering.

## Artifacts

- `outputs/analysis/r210_f1_neck_candidate_gate_smoke_val_summary.json`
- `outputs/analysis/r210_f1_neck_candidate_gate_smoke_val_per_image.csv`
- `outputs/analysis/r210_f1_neck_candidate_gate_val32_summary.json`
- `outputs/analysis/r210_f1_neck_candidate_gate_val32_per_image.csv`
- `outputs/bridge_logs/r210_f1_neck_candidate_gate_val32.log`

## Smoke Result: 8 Val Images

The 8-image smoke passed the gate.

Best setting:

- `dist_percentile`: `10`
- accepted image rate: `1.000000`
- accepted cut pixels: `213.125`
- Dice delta: `-0.000281`
- IoU delta: `-0.000498`
- Recall delta: `-0.002213`
- gap-region FP delta: `-0.004092`
- component merge rate delta: `-0.250000`
- component count MAE delta: `-0.125000`
- Boundary F1 delta: `+0.000066`
- Boundary IoU delta: `-0.000010`

Interpretation:

- Unlike R210-F0, R210-F1 produces nonzero accepted edits without component-count explosion.
- The improvement is mainly anatomy-diagnostic: gap FP and component merge/count improve while overlap remains within tolerance.
- Boundary metrics are roughly flat, not clearly improved.

## Val32 Result

A 32-image original-val subset was run because full-val with the current unoptimized candidate generator is slow.

Best setting:

- `dist_percentile`: `10`
- `num_evaluated`: `32`
- `val_gate_pass`: `true`
- accepted image rate: `0.781250`
- accepted cut pixels: `103.750000`

Anchor vs R210-F1 on val32:

| Metric | Anchor | R210-F1 | Delta |
|---|---:|---:|---:|
| Dice | 0.887218 | 0.886975 | -0.000243 |
| IoU | 0.802023 | 0.801614 | -0.000409 |
| Recall | 0.890801 | 0.889607 | -0.001194 |
| Boundary IoU | 0.232447 | 0.232368 | -0.000078 |
| Boundary F1 | 0.370793 | 0.370646 | -0.000147 |
| Surface Dice 2px | 0.650622 | 0.649967 | -0.000656 |
| Surface Dice 5px | 0.903957 | 0.903903 | -0.000053 |
| HD95 px | 17.316634 | 17.333539 | +0.016904 |
| ASSD px | 4.566438 | 4.568103 | +0.001664 |
| gap-region FP rate | 0.188887 | 0.187125 | -0.001762 |
| component merge rate | 0.375000 | 0.187500 | -0.187500 |
| component count MAE | 2.312500 | 2.156250 | -0.156250 |

Interpretation:

- R210-F1 achieves the desired direction for gap FP and component diagnostics on val32.
- Dice/IoU/Recall remain within the predefined safety tolerances.
- Boundary/surface metrics do not improve; they are essentially flat to slightly worse.
- This is promising but not enough for a final claim.

## Runtime Note

The full 96-image val run was started but stopped because the unoptimized candidate generator is slow on some large/complex masks. The bottleneck appears to be deterministic candidate generation and repeated per-candidate component analysis. A val32 run was completed to avoid spending too much time before deciding whether the route is worth optimizing.

## Decision

R210-F1 is more promising than R210-F0 and should continue.

Do not apply clean-test-v2 yet.

Next step:

1. Optimize the R210-F1 candidate generator and component analysis so full val is practical.
2. Run full original val with the locked `dist_percentile=10` setting.
3. Generate hard-case visualizations for accepted/rejected edits.
4. Only if full val passes and visualizations show no erosion/fragmentation should clean-test-v2 be considered for one locked evaluation.

Risk:

- The current gains are mostly component/gap diagnostics, not boundary/surface metrics.
- If full val preserves the val32 trend, R210-F1 can support the project objective on anatomy consistency but still needs boundary-specific refinement or a cleaner split-map model for boundary gains.
