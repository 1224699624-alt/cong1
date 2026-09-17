# R192 MaskDINO Val Prediction Diagnostic

R192 diagnosed R191 validation predictions on original val only. No training was run locally and no clean-test-v2 data was used.

## Key Numbers

- val images: `96`
- predictions: `7680`
- predictions per image: mean `80.0`
- score mean / max: `0.159449` / `0.462778`
- top-10 best bbox IoU mean: `0.182953`
- best threshold-union Dice: `0.045190` at threshold `0.1`
- best top-k union Dice: `0.045190` at top-k `80`

## Interpretation

R191 AP=0 is not just a COCO threshold artifact. It emits predictions and nonzero scores, but spatial overlap is extremely weak. The route should not touch clean-test-v2. The next valid step is a server-only overfit gate, preferably with official pretrained MaskDINO weights.
