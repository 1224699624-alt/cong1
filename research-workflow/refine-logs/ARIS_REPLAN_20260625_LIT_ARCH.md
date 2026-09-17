# ARIS Replan 2026-06-25: Literature-Driven Architecture Pivot

This replan supersedes the audit/layout-first continuation after the user clarified the desired pivot.

Primary target remains clean-test-v2 Dice `> 0.9317660066557425`.

## Hard Evaluation Rule

Use `TSRS_RSNA-Epiphysis_clean_test_v2/test` as the only target test split. Do not use the new reannotated test split for success claims.

## Pivot Decision

R064 showed that existing candidate masks cannot reach target even with a GT-leaking oracle. R055/R057 showed direct structured SAM/MedSAM adaptation is high-recall but low-precision and far below R038.

Therefore R065 should be a new deployable architecture:

1. Compact context segmenter inspired by U-Mamba / VM-UNet.
2. Boundary and separation-aware topology supervision inspired by clDice-style topology losses, adapted to separated epiphysis components.
3. SAMed/Med-SA-style adapter only as fallback, with parameter-efficient tuning and no full heavy image-encoder path.

## Run Policy

- One GPU, serial.
- Train on original epiphysis train/val only unless a later explicit data variant is approved.
- Evaluate clean-test-v2 and original test.
- No candidate-stack voting, no selector rerun, no residual editor over R038/R025b/R063.
