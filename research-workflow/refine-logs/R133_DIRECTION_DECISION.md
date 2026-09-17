# R133 Direction Decision

## Status After R132

Target remains clean-test-v2 Dice `> 0.9317660066557425`.

Current valid best is R110:

| Run | clean-test-v2 Dice | Precision | Recall | Boundary IoU | False-bridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| R110 best valid | `0.917723` | `0.913343` | `0.923580` | `0.251896` | n/a |
| R129 lightweight set/query | `0.840953` | `0.744594` | `0.970739` | `0.096504` | `0.851852` |
| R130 bridge-suppressed DINOv3 source | `0.890938` | `0.855765` | `0.931193` | `0.180672` | `0.444444` |
| R132 DINOv3 query/no-object | `0.873978` | `0.814021` | `0.945500` | `0.142318` | `0.851852` |

R132 improved over R129 but still failed with the same broad pattern: recall is high, precision is weak, component-count error is large, and bridge rate is high. Therefore the problem is not solved by adding a pretrained DINOv3/FPN encoder and query objectness to the lightweight Hungarian query model.

## Literature Check

Relevant mechanisms:

- Mask2Former uses masked attention to constrain cross-attention within predicted mask regions and is not just a set of query mask heads on an FPN feature map.
- Mask DINO extends a DETR/DINO detection system with segmentation masks, query embeddings, denoising/object-detection training, and high-resolution pixel embeddings.
- HoVer-Net-style separation is explicitly useful for clustered instance separation, but R068/R117/R130 show that this supervision alone has not transferred into a competitive epiphysis source.
- SAM/SAM2 medical adapters are promising generally, but prior project runs R055/R057/R115 showed direct SAM-style adaptation under this codebase collapsed or undersegmented.

Interpretation: R132 is not a faithful Mask2Former/Mask DINO implementation. Its failure closes lightweight query-head approximations, not the full transformer instance-segmentation family. A full Mask2Former/Mask DINO import would be a substantial engineering branch with detectron2/MMDetection-style data conversion, box/instance supervision, and careful validation. That may be scientifically reasonable, but it is no longer a quick ARIS iteration.

## Decision

Do not launch R133 as another GPU training run immediately.

R133 should be a non-GPU protocol-and-feasibility step:

1. Build a hard-case correction queue from R122/R125/R127, emphasizing the R110 clean-test-v2 cases and matched train-side analogues.
2. Separate likely label/protocol ambiguity from true model misses by producing side-by-side panels and a small editable manifest.
3. Decide between:
   - R134-data: isolated corrected/verified hard-case train/val variant, no mutation of original data.
   - R134-arch: full external Mask2Former/Mask DINO integration branch, only if the engineering cost is accepted.

Recommended next action: R133-label-protocol-manifest, because all lightweight architecture routes since R065 have failed to approach R110, while audits R122-R128 show the remaining errors are localized, mixed, and not learned cleanly from the current original-val proxy.
