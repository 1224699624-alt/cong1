# R160 HF UPerNet-ConvNeXt Smoke Summary

## Route

R160 tested a second no-custom-CUDA pretrained semantic segmentation route after SegFormer failed its gates.

- model: `openmmlab/upernet-convnext-tiny`
- implementation: Hugging Face `UperNetForSemanticSegmentation`
- dependency path: isolated `.tmp/r144_hf_deps`
- no Detectron2, mmcv, custom CUDA op, or `nvcc`

This route is literature-aligned with UPerNet-style pyramid aggregation and ConvNeXt image features, but remains a semantic segmentation source model rather than an instance/layout model.

## Smoke Protocol

- train: 16 original train images
- val selection: 8 original val images
- final diagnostic: 4 clean-test-v2 images after fixed smoke protocol
- original-test control: same 4-image smoke size
- clean-test-v2 was not used for tuning

## Result

Best validation epoch was epoch 6:

- val Dice `0.497092`
- val Precision `0.336295`
- val Recall `0.984756`
- val Specificity `0.943247`
- val Boundary IoU `0.040059`
- val false-bridge flag `1.0`

Four-image clean-test-v2 diagnostic:

- Dice `0.450227`
- Precision `0.292873`
- Recall `0.980558`
- Specificity `0.951467`
- Boundary IoU `0.017133`
- component-count error `11.5`
- false-bridge flag `1.0`

## Decision

Do not launch a full R160 run.

R160 is better than R157/R158 in raw smoke Dice, but still far below any useful source-model band and remains high-recall / low-precision with bridge-heavy masks and near-zero boundary quality. It does not justify spending GPU on longer UPerNet/ConvNeXt training.

## Next Action

Close the current Hugging Face semantic-segmentation source branch:

- HF SegFormer-B0: weak smoke and weak tiny gate.
- HF UPerNet-ConvNeXt: improved but still weak smoke.

Further progress likely requires one of:

1. an external CUDA-toolkit-capable environment for faithful MaskDINO/Mask2Former;
2. genuinely corrected train/val label PNGs and a rebuilt isolated variant;
3. a materially different no-custom-CUDA route with a stricter train-overfit gate before any clean-test-v2 diagnostic.

