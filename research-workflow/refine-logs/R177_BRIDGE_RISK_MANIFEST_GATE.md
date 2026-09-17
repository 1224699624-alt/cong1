# R177 Bridge-Risk Manifest Gate

**Date**: 2026-07-03

**Purpose**: Verify that original train/val contain enough non-leaking bridge/boundary risk signal to justify a boundary-preserving separation arbitrator.

## Artifacts

- Prep script: `scripts/prepare_r177_bridge_risk_manifest.py`
- Sync summary: `outputs/analysis/r177_trainval_anchor_sync_summary.json`
- Manifest CSV: `outputs/analysis/r177_bridge_risk_manifest/r177_bridge_risk_manifest.csv`
- Summary JSON: `outputs/analysis/r177_bridge_risk_manifest/r177_bridge_risk_manifest_summary.json`

## Data Used

- Dataset: `TSRS_RSNA-Epiphysis`
- Splits: original `train` and `val`
- Anchor proxy: `r100_like_r025b_r097_patch_basic_trainval`
- Train/val labels: original labels only
- clean-test-v2: not used for gate selection
- new/reannotated test: not used

Synchronized anchor proxy masks:

| Split | Label count | Anchor mask count |
| --- | ---: | ---: |
| train | `875` | `875` |
| val | `96` | `96` |

## Gate Result

R177-prep gate passed.

| Split | Records | Risk images | False-bridge images | Dice mean | Boundary IoU mean | Sep-band FP mean | Component error mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `875` | `276` | `276` | `0.910115` | `0.242356` | `0.190955` | `3.056000` |
| val | `96` | `32` | `32` | `0.895547` | `0.222991` | `0.190222` | `3.072917` |

Gate thresholds:

- train risk images required: `>=50`; observed `276`
- val risk images required: `>=10`; observed `32`
- missing anchor masks required: `0`; observed `0`

## Interpretation

The train/val proxy has enough bridge-risk signal for a supervised local edit model. The risk cases are not only low-Dice failures; several top-risk images still have Dice around `0.90+` but poor Boundary IoU and negative component delta. That matches the target failure: masks can look region-accurate while still merging neighboring epiphysis components.

This supports R177 as a valid next GPU branch. It also clarifies the training objective:

- do not train a full image segmenter;
- do not optimize only Dice;
- focus the model on local foreground deletion/addition in risk bands;
- select on validation with a multi-metric gate.

## Next Implementation

Create R177 local arbitrator training/evaluation:

1. Use the manifest to sample bridge-risk and non-risk patches.
2. Train on original train split using the anchor proxy and GT.
3. Select on original val split by:
   - Dice non-drop versus anchor;
   - Boundary IoU/F1 improvement;
   - false-bridge/component error reduction;
   - separation-band FP non-regression.
4. Apply the locked selected config to R110 clean-test-v2 masks once.

R177 should run on the server, one GPU at a time. Polling cadence for long training remains half-hour; short prep scripts can be run foreground.
