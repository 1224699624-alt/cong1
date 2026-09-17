# R258B Experiment Code Review

## Round 1: request changes

Independent review blocked deployment for three reasons:

1. OOF and frozen-validation proposal provenance was not hard-validated.
2. A non-limited run could change preregistered hyperparameters or use arbitrary/repeated seeds.
3. Heat Dice excluded invalid positive targets without reporting or gating heat-target coverage.

## Fixes

- Pinned and reproduced SHA256 values for train/validation proposals, fold/inference manifests, and train/validation metadata.
- Required the authorizing Phase A and R257b decisions, 875-image exact-once manifests, five folds, proposal stem-to-fold agreement, and 96 validation proposal identities.
- Enforced the exact full configuration and exact ordered seeds `2581,2582,2583`; the full launcher now supplies all fixed parameters explicitly.
- Added positive/valid heat-target counts and coverage; invalid positive targets count as Dice zero; coverage must be at least `0.85`.
- Added recursive finite-value checks and proposal rank/fold fields to the pair manifest.

## Round 2: approved for remote sanity

The second independent review found no remaining deployment blocker. Non-blocking follow-ups are: result JSON files themselves are not hash-pinned, pair audit is not yet fold-stratified, and the synthetic test does not exercise every provenance/full-gate branch. These do not authorize bypassing the remote sanity stage.
