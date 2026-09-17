# R259 Frozen-Prior nnU-Net Visual Probe

- Scope: `TSRS_RSNA-Epiphysis_contrast_v1` train/original-val only.
- Purpose: qualitative engineering probe requested by the user; no clean-test-v2, model selection, or threshold search.
- Prior: frozen ensemble of the three R258B `image_centers_age_sex` fixed-epoch checkpoints.
- Prior inputs: X-ray, continuous bone age, sex, and prediction-derived center/basin-scale proposals only. GT is prohibited during prior-map inference.
- Full-map construction: each proposal connects to its four nearest local proposals; pair probability times predicted separation heat is mapped back to the radiograph; overlapping values use pixelwise maximum and three seeds are averaged.
- nnU-Net input: channel 0 contrast-normalized X-ray; channel 1 frozen prior heatmap.
- Shared warmup: two epochs, 25 iterations/epoch, prior channel zeroed.
- Matched fork: identical full checkpoint including optimizer/AMP/logger state.
- Native branch: two additional epochs with prior channel zeroed and ordinary nnU-Net loss.
- Improved branch: two additional epochs with the prior channel plus ordinary loss and fixed `0.05` background separation loss. The auxiliary loss is evaluated only on binary-label background pixels and uses the transformed prior channel, so it remains spatially aligned after augmentation.
- Fixed inference: `checkpoint_final.pth`, original-val 96 cases. Native inference receives a zero second channel; improved inference receives the frozen prior channel.
- Visualization: cases selected by the previously frozen R255 baseline-only 1--4 px hard-case ranking, never by R259 improvement. Columns: original X-ray, GT, native nnU-Net, frozen-prior nnU-Net; both full image and close-gap crop.

This probe cannot establish a paper claim or unlock clean-test-v2.
