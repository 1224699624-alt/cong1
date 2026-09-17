# R167 Next Material Gate

## Current State

Target remains clean-test-v2 Dice `> 0.9317660066557425`.

Current valid best remains R110:

- clean-test-v2 Dice: `0.9177231563529792`
- remaining target gap: `0.014043`

R166 closed the filtered reannotated branch as a useful source or fusion candidate:

- R165 clean-test-v2 Dice: `0.890303`
- R165 better than R110 on only `2/81` images
- R110/R165 per-image oracle Dice: `0.917787`
- gain over R110: `+0.000064`

## Closed Paths

Do not launch another run in these families without new evidence:

- R165 threshold/search/fusion/readout.
- Raw or automatically filtered reannotated train/val retraining.
- DINOv3 instance-separation loss-weight sweeps.
- Lightweight HF Mask2Former from the R144-R151 setup.
- SegFormer/UPerNet/ASPP no-custom-CUDA semantic-source smoke branches.
- Same-family patch/CNN readouts over already tested weak sources.

## Available Progress Routes

### Route A: Manual Label-Protocol Reconciliation

This is the most actionable route under the current environment.

Use train/val only. Do not use reannotated test and do not use clean-test-v2 for training or tuning.

Candidate inputs already exist:

- R163 paired original-vs-reannotated train/val shift audit:
  - `outputs/analysis/r163_reannotated_pair_shift.json`
  - train same-name Dice mean `0.862338`
  - val same-name Dice mean `0.945487`
  - train has severe empty-mask and area-inflated outliers
- R125 hard-case train/val curation:
  - `outputs/analysis/r125_hard_case_curation_manifest/r125_hard_case_curation_manifest.json`
- R134 review pipeline:
  - `outputs/analysis/r134_label_protocol_review_manifest/`

Next concrete step:

1. Build an R168 review package from train/val cases where original and reannotated labels disagree most.
2. Show original image plus original-label overlay plus reannotated-label overlay plus difference overlay.
3. Let the human choose one of:
   - original label is correct
   - reannotated label is correct
   - both wrong, needs corrected PNG
   - exclude from train variant
   - uncertain / second review
4. Only after enough decisions or corrected PNGs exist, build a new isolated dataset variant.
5. Train one guarded source model on that isolated variant and evaluate success only on clean-test-v2/test.

This route has the best expected value because R163 shows the reannotated train split contains real protocol shifts, while R162/R165 show automated filtering is not enough.

### Route B: Faithful MaskDINO / Mask2Former Environment

This route is scientifically attractive but currently blocked by the server environment.

R153/R159 found:

- PyTorch `2.5.1+cu118`
- two RTX 4090 GPUs
- gcc/g++ available
- GitHub reachable
- no visible `nvcc`
- no conda/mamba/micromamba on shell PATH
- no pip-visible Detectron2 wheel for the active stack

Required before reopening:

- isolated environment, not `yolo-sam-gpu`
- CUDA toolkit and `nvcc`
- Detectron2/fvcore/iopath/pycocotools/ninja import smoke
- official MaskDINO or Mask2Former tiny train/infer smoke

Do not spend GPU on this route until those prerequisites are provided.

### Route C: Another No-Custom-CUDA Architecture

This is possible but low expected value unless it has a materially different training signal.

Minimum gate before any full run:

- one-image or small-subset overfit must pass a predeclared Dice/Bounary-IoU gate;
- validation must not show full-foreground, bridge-heavy, or empty-mask collapse;
- clean-test-v2 can only be evaluated after the protocol is fixed.

Given R143/R155/R157/R160, this route should not be the next automatic action unless Route A is impossible.

## Decision

Proceed to R168 as a train/val-only manual label-protocol reconciliation package.

Do not launch a new GPU training run yet. The next GPU run should require new material:

- corrected or explicitly selected train/val labels, or
- a newly provisioned faithful external architecture environment.

## R168 Specification

R168 should be non-GPU and local:

- Read `outputs/analysis/r163_reannotated_pair_shift.json`.
- Select approximately 80 train cases and 16 val cases with highest label disagreement.
- Include priority groups:
  - empty reannotated masks;
  - large area inflation;
  - large area shrinkage;
  - large component-count shift;
  - low original-vs-reannotated Dice.
- Generate review panels and CSV under `outputs/analysis/r168_reannotation_protocol_review_package/`.
- Keep clean-test-v2 absent from the review worklist.
- Write a machine-readable summary and update ARIS logs.

