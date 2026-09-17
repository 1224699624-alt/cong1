# R211 GT-Free Acceptance Gate Smoke Result

Date: 2026-07-07

## Purpose

Start converting R210-F1 from a GT-gated feasibility method into an inference-usable method. R211 reuses the R210-F1 deterministic neck candidates, but the accept/reject decision uses only inference-time features:

- candidate geometry,
- anchor connected-component geometry,
- split-count effect on the anchor prediction,
- cut shape,
- local image-gradient support,
- distance-transform statistics.

Ground truth is used only after prediction for R201 metric evaluation. `clean-test-v2` was not used.

## Script

- `scripts/run_r211_gt_free_acceptance_gate.py`

The script writes:

- predicted masks under isolated `outputs/ablations_variants/<output-exp>/TSRS_RSNA-Epiphysis/val/masks`
- summary JSON
- per-image CSV
- optional candidate-feature CSV

It records missing anchor masks explicitly and can skip them with `--skip-missing-anchor`; this is useful locally because only a subset of R110 trainval masks is currently synced.

## Smoke Runs

### R211-F0 strict gate

Command shape:

```bash
python scripts/run_r211_gt_free_acceptance_gate.py \
  --limit 32 \
  --skip-missing-anchor \
  --candidate-csv outputs/analysis/r211_gt_free_acceptance_gate_probe32_candidates.csv \
  --output-exp r211_gt_free_acceptance_gate_probe32 \
  --summary-json outputs/analysis/r211_gt_free_acceptance_gate_probe32_val_summary.json \
  --per-image-csv outputs/analysis/r211_gt_free_acceptance_gate_probe32_val_per_image.csv
```

Result:

- evaluated local synced cases: `4`
- missing local anchors: `28`
- accepted rate: `0.0`
- gate pass: `false`

Interpretation:

The strict rule became a no-op. Candidate-feature export showed the main blocker was not candidate absence. The generated cut candidates had strong shape and gradient support, but most did not create two large post-cut pieces. Therefore the `split_gain` / `piece_area` requirement is too strict for R210-F1-style edge/neck cleanup.

### R211-F0 relaxed rule probe

Command shape:

```bash
python scripts/run_r211_gt_free_acceptance_gate.py \
  --limit 32 \
  --skip-missing-anchor \
  --min-split-gain 0 \
  --max-split-gain 8 \
  --min-piece-area 0 \
  --min-piece-frac 0 \
  --min-gradient-ratio 0.9 \
  --max-cut-frac-component 0.02 \
  --max-cut-frac 0.004 \
  --candidate-csv outputs/analysis/r211_gt_free_acceptance_gate_relaxed2_smoke32_candidates.csv \
  --output-exp r211_gt_free_acceptance_gate_relaxed2_smoke32 \
  --summary-json outputs/analysis/r211_gt_free_acceptance_gate_relaxed2_smoke32_val_summary.json \
  --per-image-csv outputs/analysis/r211_gt_free_acceptance_gate_relaxed2_smoke32_val_per_image.csv
```

Result on the same local synced subset:

| Metric | Delta vs R110 val anchor |
|---|---:|
| accepted rate | `1.000000` |
| accepted cut pixels | `237.000000` |
| Dice | `-0.000930` |
| IoU | `-0.001555` |
| Recall | `-0.002642` |
| Precision | `+0.000673` |
| Boundary IoU | `-0.000824` |
| Boundary F1 | `-0.001171` |
| Surface Dice 2px | `-0.001913` |
| Surface Dice 5px | `-0.001963` |
| HD95 px | `+0.185657` |
| ASSD px | `+0.012939` |
| gap-region FP rate | `-0.002582` |
| component merge rate | `-0.500000` |
| component count MAE | `-1.000000` |

Interpretation:

The relaxed GT-free rule recovers the expected anatomy-consistency direction: lower gap FP, lower component merge rate, and lower component count MAE. However, the same run worsens recall, boundary metrics, Surface Dice, HD95, and ASSD. This is exactly the failure mode we need to prevent before clean-test-v2.

## Candidate Feature Finding

On the local synced subset, exported candidates have:

- high gradient support: median gradient ratio about `2.39`
- thin/neck-like shape: median slenderness about `5.5`
- small component cut fraction: median about `0.0089`
- low distance-transform ratio: median about `0.14`

But most candidates have:

- `split_gain=0`
- `min_piece_area=0`

Therefore, requiring a candidate to immediately split the anchor into two large pieces is not compatible with the useful R210-F1 correction pattern. The candidate generator often performs edge/neck cleanup rather than a full connected-component split.

## Decision

R211-F0 is a useful implementation step but not a passed model gate.

Current conclusion:

- A purely strict GT-free geometry rule becomes no-op.
- A relaxed GT-free geometry rule makes nonzero anatomy-consistency improvements but reintroduces recall/boundary risk.
- The next R211 step should not be another manual threshold tweak. It should train or calibrate a GT-free acceptance classifier on original train/val using R210-F1 candidate features and GT-derived labels only during training.

## Next Step: R211-F1 Learned Acceptance Classifier

Build a train/val-only candidate table:

- Generate R210-F1 candidates on train and val.
- For each candidate, compute inference-available features only.
- Temporarily evaluate candidate impact with GT on train/val to create training labels:
  - positive if gap/component metrics improve and Dice/Recall/Boundary do not exceed tolerance,
  - negative if the candidate causes recall/boundary loss or fragments anatomy.
- Fit a small interpretable classifier or calibrated score.
- Lock the classifier and threshold using val only.
- Apply to full original val and evaluate with R201.

Clean-test-v2 remains locked until R211-F1 passes full original-val R201 metrics and hard-case visual audit.
