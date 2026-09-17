# R207 Baseline Coverage Audit

Date: 2026-07-06

## Purpose

Audit which baselines are already ready under the frozen R201 protocol and which require mask sync/regeneration before they can be used in full diagnostic tables.

## Artifacts

- Script: `scripts/build_r207_baseline_coverage_audit.py`
- JSON: `outputs/analysis/r207_baseline_coverage_audit.json`
- CSV: `outputs/analysis/r207_baseline_coverage_audit.csv`

## Ready For R201 Tables

These have full mask-derived R201 metrics with `num_evaluated=81`:

- ARAA DANet epoch98
- R110 current best
- R202 nnU-Net 2D risk-audit baseline
- R143 high-res medical U-Net recipe

Key reminder:

- R202 has the strongest overlap/boundary metrics but is not the main target to beat on Dice/IoU.
- R143 is already an included U-Net-style baseline and is much weaker than R110/R202.

## Candidate Baselines Requiring Mask Backfill

These have legacy clean-test-v2 metrics but no local mask directory for R201 full recomputation:

- R106 Swin tiny U-Net:
  - Dice `0.836148`
  - Boundary IoU `0.101564`
  - Recommendation: sync/regenerate masks only if a Swin-family baseline is explicitly needed in the comparison table.
- R097 ConvNeXt tiny U-Net:
  - Dice `0.867598`
  - Boundary IoU `0.137583`
  - Recommendation: sync/regenerate masks only if an additional U-Net-like baseline is needed.

Both are far weaker than R110 and R143, so they should not delay the main project-improvement loop.

## Diagnostic-Only Failed Repair Branches

Do not backfill masks for these unless needed for a very specific failure analysis:

- R177 boundary-preserving arbitrator
- R179 boundary/gap constrained source
- R184 guarded no-op
- R186 fast neck gate

These branches already have enough evidence to support the decision that local pixel deletion/bridge editing either fragments anatomy or becomes a no-op.

## Decision

The baseline table is already strong enough for the current project phase:

- ARAA as the domain baseline.
- R110 as the internal anchor.
- R143 as an existing U-Net-style medical recipe baseline.
- R202 nnU-Net as internal risk-audit/upper-bound context.

Next work should not be blocked on R106/R097 backfill. If the user wants a named Swin baseline in the final comparison, backfill R106 masks or train a cleaner Swin-UNet/SwinUNETR baseline later.
