# R158 HF SegFormer Gate Summary

## Route

R156 allowed one no-compile pretrained/literature probe after R155 ASPP collapsed. The selected route was Hugging Face SegFormer-B0:

- paper family: SegFormer, hierarchical Transformer encoder plus lightweight MLP decoder
- implementation: `SegformerForSemanticSegmentation`
- dependency path: isolated `.tmp/r144_hf_deps` with `transformers==4.53.0`
- no Detectron2, mmcv, custom CUDA op, or `nvcc` required

## R157 Smoke

R157 trained on 16 original-train images, selected on 8 original-val images, and evaluated only 4 clean-test-v2 images after the fixed smoke protocol.

Best validation Dice was `0.374114`.

Four-image clean-test-v2 diagnostic:

- Dice `0.314837`
- Precision `0.193409`
- Recall `0.904590`
- Specificity `0.918213`
- Boundary IoU `0.005224`
- false-bridge flag `1.0`

This avoided R155's full-foreground collapse, but remained far below any useful source-model band.

## R158 Tiny Gate

R158 used 1 train image and 1 validation image and skipped clean-test-v2 final evaluation.

It early-stopped at epoch 22. Best one-val Dice was only `0.492192` at epoch 7.

This is not a strict same-image train-overfit proof because the validation image was different from the train image. However, it is a useful tiny-transfer gate: if SegFormer-B0 were promising under this loss/readout, a 1-train/1-val gate should show a much stronger signal than `0.49`.

## Decision

Do not launch a full HF SegFormer run from the current recipe.

Reasons:

- R157 smoke clean-test-v2 diagnostic is only `0.314837`.
- R158 tiny gate peaks at only `0.492192`.
- Boundary IoU remains near zero.
- Failure mode remains low-precision, bridge-heavy prediction.
- Current clean-test-v2 target is `> 0.9317660066557425`; current valid best remains R110 at `0.9177231563529792`.

## Next Action

Pause local lightweight architecture launches. The remaining viable options are:

1. build an isolated CUDA-toolkit-capable environment and retry faithful MaskDINO/Mask2Former;
2. obtain or produce genuinely corrected train/val label PNGs and rebuild an isolated data variant;
3. perform a new literature search for a materially different no-compile pretrained model, but only if it has a strict train-overfit gate before any clean-test-v2 diagnostic.

