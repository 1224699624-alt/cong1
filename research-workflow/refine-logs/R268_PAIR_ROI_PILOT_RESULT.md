# R268 pair-ROI pilot result

## Scope

- Dataset: `TSRS_RSNA-Epiphysis` original-val only
- Images: 96
- Anchor: train/val R110 masks
- clean-test-v2: not used
- GPU training: not run because the local PyTorch installation is CPU-only

## Implementation

`scripts/run_r268_pair_roi_pilot.py` builds generic adjacent-pair ROIs from the
anchor connected components and X-ray center-line valley evidence. Labels are
opened only after proposal generation for coverage audit; no labels select or
modify predictions.

## Result

- mean ROIs per image: `7.9583`
- mean maximum single-ROI GT-gap coverage: `0.1359`
- mean union coverage of all candidate ROIs over the GT gap band: `0.3501`

The overlay review shows that the current nearest-component proposal rule often
places boxes near phalangeal/MCP boundaries and does not reliably cover merged
metacarpal or carpal bridges. Therefore this is a coverage failure, not evidence
that the generic pair-loss is ineffective.

## Decision

Do not train R267/R268 from this proposal generator. The next ROI generator must
add a merge-risk detector inside a single connected component (neck/basin or
distance-transform saddle), while retaining the current pair ROI for already
separated neighbors. The training/inference contract remains generic gap/support
maps with abstention; no CC/CR labels are required.
