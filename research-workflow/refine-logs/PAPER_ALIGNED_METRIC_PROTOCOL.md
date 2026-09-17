# Paper-Aligned Segmentation Metric Protocol

## Primary metrics

Use the RAM-W600 benchmark definitions implemented with MONAI 1.4.0:

- DSC (higher is better).
- NSD at a fixed 2-pixel tolerance (higher is better).
- VOE = 1 - IoU (lower is better).
- Symmetric MSD in pixels (lower is better).
- RAVD = absolute area/volume difference divided by GT area/volume (lower is better).

RAM-W600 overlap evaluation uses the same metrics on the logical intersection of each
predefined bone pair. TSRS applies the same metrics to a documented, high-confidence local
gap ROI. The metric formulas are standard; only the anatomical ROI is task-specific.

For TSRS pair-crops without any true background gap (absent/contact/uncertain relation), do
not mix empty targets into the segmentation mean. Evaluate gap presence with the RAM-W600
paper's standard classification family: balanced accuracy, F1, sensitivity, specificity,
precision and accuracy. Compute DSC/NSD/VOE/MSD/RAVD only on crops with a non-empty GT gap.

Never clip a predicted gap/overlap mask with a GT-derived interface ROI before computing the
paper-facing metric. GT-clipped predictions are oracle diagnostics only. Paper-facing metrics
compare the complete prediction support inside the declared inference unit (full image or full
pair crop) against the GT target, matching RAM-W600's official overlap-mask evaluation logic.

## Reporting hierarchy

1. Paper main tables: DSC, NSD@2px, VOE, MSD; RAVD in main or supplementary table.
2. Per-interface tables: per-pair DSC and NSD@2px; VOE/MSD/RAVD may be supplementary.
3. Training diagnostics only: precision, recall, soft Dice and loss dynamics.
4. Internal failure analysis only: gap-region FP, component merge rate, component count MAE,
   Boundary IoU/F1, NSD@5px and HD95 unless a separate cited protocol explicitly requires them.

## Scale and protocol separation

- RAM-W600 paper-aligned evaluation uses 512x512 inputs and NSD tolerance 2 px.
- R289 gap-head evaluation uses native-scale 512x512 local crops and tolerance 2 px.
- Development validation must include every selected pair that is actually present and adjacent
  in each case. Truncating to the first K pairs is forbidden because it biases results toward
  common mature structures and under-represents age-dependent ossification centers.
- Frozen R201 original-resolution results remain valid historical results. R201 now emits
  additive aliases (`dsc`, `nsd_2px`, `voe`, `msd_px`, `ravd`) without removing or changing
  legacy fields. Do not directly compare 2-pixel NSD values across different resize policies.
- `clean-test-v2` remains final-only and is never used for threshold or model selection.

## Reproducibility

Random-seed replication is deferred by current instruction. Single-run development results
must be labelled as pilots and not reported as final mean +/- standard deviation claims.
