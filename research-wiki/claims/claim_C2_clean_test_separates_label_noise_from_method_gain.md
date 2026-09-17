---
type: claim
node_id: claim:C2_clean_test_separates_label_noise_from_method_gain
status: supported
added: 2026-06-15T12:55:00Z
---

# Claim: clean-test-v2 helps separate label-noise effects from method effects

## Statement
A secondary evaluation slice that excludes a small set of suspicious test annotations is useful for estimating how much reported performance is influenced by label quality rather than model quality.

## Evidence

- `clean-test-v2` excludes suspicious test cases without touching the original dataset
- both ARAA and current refiner results have already been compared on this slice
- the resulting Dice values are higher than noisier original test evaluations

## Current status

Supported as an evaluation practice in this project.

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
