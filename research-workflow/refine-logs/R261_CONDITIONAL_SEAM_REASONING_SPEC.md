# R261 Age/Sex-Conditional Per-Seam Reasoning

## Scope and protocol

- Dataset: original `TSRS_RSNA-Epiphysis` train/original-val only.
- `TSRS_RSNA-Articular-Surface` and `clean-test-v2` are rejected by path guards.
- Original data layout remains unchanged; all maps, nnU-Net data, checkpoints, masks, and reports use isolated R261 paths.
- Baseline initialization is the mature R202 nnU-Net checkpoint. Newly added prior channels are zero-initialized.

## Method

1. Build a train-label-only relation bank from instance centers and scales in a per-image canonical coordinate frame.
2. Condition the bank continuously on bone age and sex using age kernels, sex-aware backoff, and per-case balancing.
3. Generate GT-free candidate centers from the existing OOF train proposals and frozen image-only original-val proposals.
4. Score pairs in two stages: conditional relation likelihood, then X-ray valley evidence plus proposal confidence.
5. Keep at most six independent seam slots. Each slot has an anisotropic seam stripe and two bone-support shoulder maps; uncertain slots remain zero (full-image fallback).
6. Feed the 12 maps as auxiliary nnU-Net channels. For every slot, optimize a confidence-weighted relation margin between seam foreground probability and support foreground probability, plus a side-preservation floor. The original binary bone mask remains the only hard segmentation target.
7. Balance relation and native nnU-Net gradients with a warm-up and a bounded gradient-ratio coefficient. Do not increase a fixed background penalty.

## Sanity gate

- Train prior uses only original train labels and train metadata.
- Original-val maps use no original-val labels.
- Exact metadata/proposal coverage and train/val identity disjointness.
- Non-finite maps, empty proposal coverage, seam/support overlap, and missing slots are reported.
- One-batch relation-loss test must show nonzero valid slots, finite gradients, and a lower relation loss after an optimizer step.
- No full training before the gate passes.

## Evaluation gate

- R201 original-val metrics versus mature R202.
- Dice/IoU and Recall are guardrails; primary targets are Boundary IoU/F1, Surface Dice 2px/5px, HD95, ASSD, gap FP, merge rate, and component-count MAE.
- Twelve frozen R255 hard-pair panels, not result-selected cases.

## Sanity outcome (2026-07-15)

- Full prior construction completed, but anatomical targeting failed before GPU training.
- Train selected slots: `5243/875`, mean `5.992/6`, abstention `0/875`.
- Original-val selected slots: `576/96`, exactly `6/6`, abstention `0/96`.
- Visual audit showed that global full-hand ranking mixes metacarpal and carpal relations. The displayed green peaks were inward-shifted support shoulders rather than proposal/bone centers, making the diagnostic overlay misleading and exposing geometry mismatch.
- Decision: `STOPPED_BEFORE_GPU`. The screen was terminated during Dataset205 copying; no nnU-Net sanity or full training ran.
- Required redesign: detect the metacarpal-head row first, pair only anatomically adjacent centers in left-right order, search each center-to-center profile for image/model-supported bone shoulders and the intervening probability/intensity valley, and permit real per-seam abstention.
