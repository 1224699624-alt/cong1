# R130 Decision Artifact

Date: 2026-06-29

Target: clean-test-v2 Dice `> 0.9317660066557425`.

Current valid best: R110 `r100_r108_patch_basic`, clean-test-v2 Dice `0.9177231563529792`.

Evaluation rule: success must use `TSRS_RSNA-Epiphysis_clean_test_v2/test` only. The reannotated test split remains excluded.

## Why R129 Closes The Lightweight Set-Matching Path

R129 tested the smallest viable variable-slot idea after R128 showed that fixed anatomical slots and single-axis ordering are not reliable. It used query masks with Hungarian matching against instance-valued labels plus a binary union head.

Result:

| Run | clean-test-v2 Dice | Precision | Recall | Boundary IoU | False bridge | Delta vs R110 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R110 | `0.917723` | `0.913343` | `0.923580` | `0.251896` | not primary | baseline |
| R117 | `0.894567` | `0.859591` | `0.934903` | `0.185712` | `0.728395` | `-0.023156` |
| R126 | `0.887837` | `0.847248` | `0.934430` | `0.166387` | `0.839506` | `-0.029886` |
| R129 | `0.840953` | `0.744594` | `0.970739` | `0.096504` | `0.851852` | `-0.076771` |

Interpretation: R129 is not a near miss. It is a high-recall / low-precision failure with severe bridge-like unions and very poor boundary quality. The likely failure is insufficient query/background calibration and weak separation structure, not a single bad threshold. A blind sweep over query count, threshold, or blend would be low value.

## Literature-Guided Direction Filter

Useful references:

- Mask2Former: masked-attention mask classification for universal segmentation, using localized cross-attention over predicted mask regions. Source: https://arxiv.org/abs/2112.01527
- Mask DINO: DETR/DINO-style query embeddings with a mask prediction branch over high-resolution pixel embeddings. Source: https://arxiv.org/abs/2206.02777
- Boundary loss: contour-distance supervision for imbalanced segmentation, complementary to regional losses. Source: https://arxiv.org/abs/1812.07032
- clDice: differentiable topology-preserving loss via skeleton overlap. Source: https://arxiv.org/abs/2003.07311

Implication for this project:

R129 borrowed only the permutation-invariant matching idea, but not the stronger parts that make query segmentation usable: transformer/masked attention, no-object/background calibration, high-resolution pixel embeddings, iterative mask refinement, and explicit objectness. Meanwhile, the actual failure mode is bridge-heavy overmasking, so any next architecture must also add separation/boundary pressure before unioning instances.

## R130 Gate

Do not launch a GPU run immediately.

R130 should first produce one of these two non-GPU deliverables:

1. Label/protocol route: manual review of R122/R125 panels and an isolated corrected/verified hard-case manifest. This is justified if visual review finds annotation ambiguity or a consistent protocol mismatch.
2. Architecture route: an implementation plan for a stronger bridge-preventing query model, not another lightweight U-Net slot head. Minimum requirements:
   - pretrained encoder retained from the stronger R100/R110 line or DINOv3 source work;
   - query/no-object calibration similar in spirit to Mask2Former/Mask DINO;
   - explicit binary union head kept separate from instance queries;
   - boundary or signed-distance supervision;
   - separation/topology penalty that discourages bridges between neighboring epiphysis components;
   - validation selection on original val only, with clean-test-v2 used once for final evaluation.

## Decision

R129 is `DONE_NEGATIVE`. The next GPU experiment is blocked until R130 chooses a concrete protocol-correction artifact or a substantially stronger query/bridge-prevention architecture. R110 remains the current valid best.
