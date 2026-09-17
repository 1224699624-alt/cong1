# R222 Seam Background Scorer Result

Date: 2026-07-08

## Purpose

Continue the R221 instance-preserving seam/background route after the inverted
topology idea showed oracle signal: restore background connectivity in true
bone-gap regions to reduce bone-seam adhesion and boundary errors without
trading for over-erosion or missed epiphysis foreground.

This is original train/val development only. It does not use clean-test-v2 and
is not R201 paper-facing evidence.

## Implemented

Added and updated:

- `scripts/train_r222_seam_background_scorer.py`

The script trains a GT-free pixel MLP from original train data:

- input: grayscale image, R110 anchor mask, gradient, distance/boundary maps,
  normalized coordinates, and local image statistics;
- target: R221-like oracle gap cut restricted to seam-eligible anchor pixels;
- validation: original val masks only;
- output: isolated R222 mask directories and JSON/CSV summaries.

The latest update also adds probability diagnostics:

- `--diagnostic-json`
- per-image eligible-region probability percentiles;
- raw threshold candidate counts;
- post-opening candidate counts;
- oracle cut size after eligible-region restriction.

## Runs

Remote workspace:

- `/home/shenzeyu/workspace/YOLO_SAM_generic_src`

Remote Python:

- `/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin/python`

Synced artifacts:

- `outputs/analysis/r222_seam_background_scorer_pilot64_summary.json`
- `outputs/analysis/r222_seam_background_scorer_pilot64_per_image.csv`
- `outputs/analysis/r222_seam_background_scorer_pilot64_lowthr_cached_summary.json`
- `outputs/analysis/r222_seam_background_scorer_pilot64_lowthr_cached_per_image.csv`
- `outputs/analysis/r222_seam_background_scorer_pilot64_lowthr_diag_summary.json`
- `outputs/analysis/r222_seam_background_scorer_pilot64_lowthr_diag_diagnostic.json`
- `outputs/analysis/r222_seam_background_scorer_pilot64_lowthr_diag_per_image.csv`
- `outputs/logs/r222_seam_background_scorer_pilot64_lowthr_diag.log`

## Key Results

Pilot64 original-val subset:

- validation images: `32`
- best threshold: `0.20`
- best max cut fraction: `0.0015`
- mean cut pixels: `0.0`
- gate pass: `false`

Low-threshold cached sweep:

- thresholds: `0.001,0.005,0.01,0.02,0.05,0.08,0.10,0.15,0.20`
- max cut fractions: `0.0015,0.003,0.006`
- grid configs: `27`
- nonzero-cut configs: `0`
- best mean cut pixels: `0.0`
- gate pass: `false`

Diagnostic sweep:

- thresholds: `1e-6,1e-5,1e-4,0.001,0.005,0.01`
- max cut fractions: `0.0015,0.003,0.006`
- grid configs: `18`
- nonzero-cut configs: `0`
- best mean cut pixels: `0.0`
- gate pass: `false`

Diagnostic mean on val32:

- eligible pixels per image: `5867.78125`
- oracle cut pixels after eligible restriction: `51.625`
- oracle cut fraction inside eligible region: `0.007461`
- eligible probability p50: `0.515031`
- eligible probability p99: `0.524091`
- eligible probability p100: `0.526303`
- raw pixels above `1e-6`: `5867.78125`
- post-opening pixels above `1e-6`: `4229.09375`
- final accepted cut pixels after top-k/opening/min-area path: `0.0`

## Interpretation

R222 failed as a deployable mask-changing scorer.

The failure is not caused by thresholds being too high. Even after lowering the
threshold to `1e-6`, the final mask edit remains zero. The diagnostic shows the
MLP assigns nearly constant probabilities around `0.515` across the eligible
region, so top-k selection is not spatially meaningful. The selected pixels are
too diffuse or not aligned with coherent seam structures, and the existing
opening/min-area safety path removes them before mask writing.

R221 remains useful oracle evidence, but R222 does not convert that oracle
signal into a GT-free spatially selective model.

## Decision

`R222 = no_go_for_mask_level`.

Do not:

- continue training the same pixel MLP for more epochs;
- tune thresholds on clean-test-v2;
- apply R222 to clean-test-v2;
- claim any R201 improvement from R222.

## What This Means for the Project

The background-connectivity idea is still alive, but the current implementation
is the wrong inductive bias. The model needs to predict coherent seam/background
structures or candidate components, not independent pixels with a shallow MLP.

The next route should preserve the R221 insight while changing the unit of
prediction:

- candidate/component-level seam background proposals;
- connected background-channel scoring;
- explicit foreground-preservation and component-count guards;
- mask-level validation only on original val before any clean-test-v2 use.

