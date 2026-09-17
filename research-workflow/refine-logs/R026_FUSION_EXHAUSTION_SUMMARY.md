# R026 Fusion Exhaustion Summary

Date: 2026-06-23

Target: clean-test-v2 Dice > 0.9317660066557425 (original ARAA 0.9117660066557425 + 0.02).

## Current Best

- R025b `r025b_anchor_local_pixel_residual_sam_hqsam`
- clean-test-v2 Dice `0.914298`
- IoU `0.842663`
- Precision `0.900455`
- Recall `0.929894`
- Boundary IoU `0.242169`

## Negative/Low-Yield Runs

- R020 mask-stack CNN fusion: Dice `0.891170`.
- R021 R015-anchored mask-stack CNN fusion: Dice `0.889575`.
- R022b R015-anchored fast bit-pattern fusion: Dice `0.914214`.
- R024 R015-anchored selector refiner v3: Dice `0.902147`.
- R025b anchor-local pixel residual MLP: Dice `0.914298`.
- R026 train+val calibrated bit-pattern fusion: Dice `0.914293`.

## Diagnosis

- Per-image oracle over available candidates is only `0.915987`, so choosing among existing masks cannot reach the target.
- Pixel oracle over available candidates is `0.957180`, so useful signal exists only as localized pixel-level edits.
- Current pixel-level calibration and local MLP can recover only tiny improvements over R015 (`~+0.0008` Dice), far short of the required additional `~+0.0175`.

## Next Direction

R027 should generate new candidate evidence rather than continue fusing the same candidate set. Recommended routes:

- hard-case targeted SAM re-prompting for images where R015 Dice is low or recall/precision is imbalanced;
- image-level quality predictor to route only hard cases into an alternate prompt/refinement pipeline;
- new error-refiner target that explicitly predicts missing/excess regions from image features and candidate disagreement, but with inference constrained to R015 local edit zones.
