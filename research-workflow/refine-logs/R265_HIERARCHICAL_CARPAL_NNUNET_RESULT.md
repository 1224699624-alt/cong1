# R265 Hierarchical Carpal nnU-Net Result

## Protocol

- Scope: TSRS_RSNA-Epiphysis original train/original-val only.
- clean-test-v2 was not read.
- Initialization: mature R202 nnU-Net checkpoint, SHA256 `65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7`.
- Prior slots: four carpal-carpal (strong), one carpal-metacarpal (medium), one carpal-radius/ulna (weak).
- Conservative abstention: train mean 3.861/6, val mean 3.750/6.
- Two-epoch GPU sanity passed; full run early-stopped at epoch 15. Best EMA checkpoint was established at the first full epoch.

## Mechanism observation

- The mature checkpoint already had low seam probability and high support probability on most selected slots.
- The margin relation loss therefore saturated quickly: epoch-0 mean gap probability 0.292, support probability 0.969, relation loss 0.00335, alpha 0.00133.
- The auxiliary gradient remained very small relative to native fine-tuning and did not create a reliable separation change.

## Original-val R201-style comparison

Relative to the matched mature nnU-Net baseline:

- Dice: +0.000302
- Recall: -0.007229
- Boundary IoU: -0.005413
- Boundary F1: -0.006091
- Surface Dice 2 px: -0.008900
- Surface Dice 5 px: -0.006856
- gap-region FP rate: +0.014439 (worse)
- component merge rate: +0.156250 (worse)
- component count MAE: -2.864583

## Decision

No-go for the current loss formulation. The four-column panels show mostly subtle changes and several enlarged/merged carpal predictions rather than the requested obvious seam recovery. Do not use clean-test-v2. The next revision must make the hard carpal-carpal term non-saturating and select checkpoints with an original-val carpal seam/boundary guardrail rather than EMA Dice alone.

An earlier visual interpretation that treated a young-child cyan CR edge as necessarily incorrect was revised after anatomical clarification. Full age-stratified counts show CR relations concentrate in younger developmental stages and vanish in the older validation groups. The no-go decision applies to the saturated loss/training result, not to the developmental-prior concept itself. See `R265_DEVELOPMENT_RELATION_SYNC_AND_ANALYSIS.md`.

Artifacts:

- `outputs/visualizations/r265_hierarchical_carpal_nnunet_original_val/`
- `outputs/analysis/r265_baseline_original_val_r201.json`
- `outputs/analysis/r265_hierarchical_carpal_original_val_r201.json`
