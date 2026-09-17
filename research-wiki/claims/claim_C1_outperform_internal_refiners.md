---
type: claim
node_id: claim:C1_outperform_internal_refiners
status: open
added: 2026-06-15T12:55:00Z
---

# Claim: New refiner directions should outperform current internal baselines

## Statement
A new refiner or refinement pipeline should improve mean Dice over the strongest current internal refiner baselines on TSRS_RSNA-Epiphysis, ideally without sacrificing boundary quality.

## Evidence needed

- controlled metric comparison on the same split
- boundary IoU comparison
- bad-case visualization on known difficult samples

## Current status

Open. Internal variants have improved substantially over early baselines, but a robust margin over the strongest ARAA comparison has not yet been established.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
