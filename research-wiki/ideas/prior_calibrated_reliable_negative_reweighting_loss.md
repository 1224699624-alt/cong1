---
type: idea
node_id: idea:prior_calibrated_reliable_negative_reweighting_loss
title: "Prior-Calibrated Reliable Negative Reweighting Loss"
status: refined
added: 2026-07-13T14:44:52+08:00
---

# Prior-Calibrated Reliable Negative Reweighting Loss

## Thesis

Use the original noisy bone mask as the only hard target. Conservative label reliability controls base-loss gradient allocation, while a frozen developmental relation prior only reweights selected hard-negative pixels that the original label already marks as background between two candidate bones.

## Core safety rules

- no inverted seam label or generated background corridor;
- no valid `y=0` anchor means no pair loss;
- model prediction chooses hard-negative priority, never target direction;
- prior produces a stop-gradient coefficient only;
- multiple pair weights use bounded max aggregation;
- pair loss is normalized by selected weight mass, not full-image background.

## Relationship

This idea supersedes the max-min bridge-energy training objective in [[heterochrony_aware_prediction_derived_separation_prior]] while retaining its frozen developmental relation posterior as a coefficient generator.

## Status

Method refinement READY at 9.27/10. No experiment was run. Canonical proposal: `research-workflow/refine-logs/FINAL_CONFIDENCE_WEIGHTED_LOSS_PROPOSAL_20260713_144452.md`.

