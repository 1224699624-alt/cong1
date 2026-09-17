# R324 RAM Explicit Overlap IEM Status

## Objective

Test an output-side explicit bone-overlap prior on RAM-W600, inspired by Zhao
et al. (MICCAI 2025), without repeating the learned global overlap-head design
used by R284.

## Experimental isolation

- Dataset: RAM-W600 only
- Split: official 425 train / 69 validation
- Test used: no
- Backbone: mature R285 multi-label nnU-Net
- Initialization: identical R285 checkpoint for both arms
- Arms: plain fine-tuning vs explicit overlap-IEM fine-tuning
- Output: `outputs/ram_w600/r324_nnunet_explicit_overlap_iem`
- Screen: `r324_overlap_iem`
- Log: `outputs/bridge_logs/r324_overlap_iem.log`

## Method

The 14 sigmoid bone-instance logits are passed to a training-only explicit
interaction module. A relation graph is computed from overlap frequencies in
RAM training masks. For every related bone pair, the module identifies:

1. spurious assignment of one bone inside the other outside genuine overlap;
2. missing membership of either bone inside genuine multi-label overlap.

These locations form a detached violation map. A local Dice+BCE penalty is
applied to the original differentiable logits on that map. The module has no
test-time GT dependency and is absent at inference.

## Coefficient audit

The rib paper reports lambda=0.3, but this is not directly transferable to the
mature R285 loss scale. On eight RAM training batches:

- median native gradient norm: 0.0710384
- median prior gradient norm: 11.3050310
- median native/prior gradient ratio: 0.00626166

R324 fixes lambda=0.001, giving an initial prior/native gradient ratio of about
0.16. This coefficient was determined from training batches only, not by a
validation sweep.

## Metrics

RAM paper-aligned metrics are reported globally and on overlapping regions:
DSC, NSD at 2 px, VOE, MSD, and MSD failure rate. Macro IoU, sensitivity and
specificity are retained as supporting diagnostics.

## Launch state

- Launched: 2026-08-03 12:45 Asia/Hong_Kong
- GPU: RTX 4080 SUPER
- First plain epoch completed successfully
- First-epoch plain macro DSC: 0.977872
- First-epoch plain overlap DSC: 0.871068
- First-epoch plain overlap NSD@2px: 0.873740

## Literature anchors

- Zhao et al., *Learning with Explicit Topological Priors for Chest X-ray Rib
  Segmentation*, MICCAI 2025.
- Wang et al., *BLS-GAN: A Deep Layer Separation Framework for Eliminating
  Bone Overlap in Conventional Radiographs*, AAAI 2025.
- You et al., *Learning With Explicit Shape Priors for Medical Image
  Segmentation*, IEEE TMI 2025.
- Wyburd et al., *Anatomically Plausible Segmentations: Explicitly Preserving
  Topology Through Prior Deformations*, Medical Image Analysis 2024.
- Tran et al., *ShapeFormer: Shape Prior Visible-to-Amodal Transformer-based
  Amodal Instance Segmentation*, 2024 preprint.
