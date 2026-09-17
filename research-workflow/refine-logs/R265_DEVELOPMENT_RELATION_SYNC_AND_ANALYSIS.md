# R265 Developmental Relation Sync and Analysis

## Correction to the visual interpretation

The cyan `CR` edge in the young-child example is anatomically plausible. In an incompletely ossified wrist, the visible distal carpal row is sparse and a detected carpal ossification center can legitimately be related to the distal radius. A fixed adult-style two-row carpal interpretation is invalid for these cases.

The relation overlay has the following meaning:

- green points: image-derived bone-side intensity peaks that should preserve foreground confidence;
- colored segment: a diagnostic pair connection, not the actual cut or loss support;
- red `CC`: carpal-carpal, strong relation;
- orange `CM`: carpal-metacarpal-base, medium relation;
- cyan `CR`: carpal-radius/ulna, weak relation;
- the actual seam map is a narrow anisotropic heatmap centered at the intensity valley between the two green supports.

## Developmental evidence from the full R265 prior set

Mean selected relations per validation image by bone age:

| Bone age | N | CC | CM | CR | All selected | Empty slots |
|---|---:|---:|---:|---:|---:|---:|
| <=60 months | 23 | 2.087 | 0.174 | 0.261 | 2.522 | 3.478 |
| 61-84 months | 16 | 3.938 | 0.062 | 0.375 | 4.375 | 1.625 |
| 85-108 months | 14 | 4.000 | 0.000 | 0.286 | 4.286 | 1.714 |
| 109-132 months | 14 | 4.000 | 0.000 | 0.000 | 4.000 | 2.000 |
| >132 months | 29 | 4.000 | 0.000 | 0.000 | 4.000 | 2.000 |

The train split shows the same trend: CR is most frequent in the 61-108 month range and falls sharply after 109 months. This supports an age-conditioned developmental-state interpretation instead of a fixed spatial row rule.

## What worked

1. The prior generator expresses changing ossification count: young cases select fewer CC edges and abstain more often.
2. CR relations occur mainly in developmentally younger wrists rather than uniformly across ages.
3. Full train/validation construction is GT-free at validation time and clean-test-v2 was not read.
4. The mature R202 checkpoint was loaded exactly; new prior channels began with zero convolution weights.
5. The selected checkpoint, predictions, priors, provenance JSON, and logs are now synchronized locally and checksum verified.

## Why nnU-Net did not improve the target behavior

The prior idea and the loss behavior must be separated. The prior carries a meaningful developmental signal, but the current margin loss is mostly already satisfied by the mature baseline:

- epoch-0 mean seam/gap probability: 0.292;
- epoch-0 mean bone-support probability: 0.969;
- CC margin: 0.18;
- therefore `0.18 + 0.292 - 0.969 < 0` for most slots, producing zero hinge gradient;
- epoch-0 relation loss was only 0.00335 and its adaptive coefficient was only 0.00133.

As a result, most optimization was ordinary nnU-Net fine-tuning. The best EMA Dice was established at the first full epoch, and training stopped at epoch 15. The auxiliary relation signal did not become strong enough to open narrow carpal seams. On original-val, Boundary IoU/F1, Surface Dice, gap FP, and component merge rate worsened even though component-count MAE improved.

Label quality further weakens the current loss because it gates seam pixels with inverse binary GT. If a true seam is mislabeled as bone foreground, that seam contributes little or no relation gradient. This is safe against destructive cuts but can make the intended supervision disappear.

## Correct next formulation

Keep the developmental prior, including valid young-child CR relations, but replace the saturated relative margin with a confidence-weighted absolute seam objective:

- strong CC: explicitly push image-supported valley probability below an age/contrast-conditioned ceiling;
- preserve both green bone supports with a foreground floor;
- CM and CR retain lower weights and type-specific ceilings;
- do not require inverse GT to define the seam; use image valley evidence and proposal confidence, while using GT only to reject obvious bone-interior contradictions;
- select checkpoints with original-val carpal gap/boundary guardrails, not EMA Dice alone.

## Local synchronized artifacts

- bundle: `outputs/artifact_bundles/r265_useful_bundle.tar.gz` (460,701,957 bytes);
- checksums: `outputs/artifact_bundles/R265_REMOTE_ARTIFACT_SHA256.txt`;
- full priors and 971 pair records: `outputs/priors/r265_hierarchical_carpal/`;
- selected checkpoint: `outputs/nnunet/r265_hierarchical_carpal/nnUNet_results/Dataset206_TSRS_RSNAEpiphysisHierarchicalCarpal2D/nnUNetTrainerR265HierarchicalCarpal__nnUNetPlans__2d/fold_0/checkpoint_best.pth`;
- 96 best-checkpoint predictions: `outputs/nnunet/r265_hierarchical_carpal/predictions_best/`;
- four-column panels: `outputs/visualizations/r265_hierarchical_carpal_nnunet_original_val/`;
- metric files: `outputs/analysis/r265_baseline_original_val_r201.json` and `outputs/analysis/r265_hierarchical_carpal_original_val_r201.json`.

Checkpoint SHA256 verified locally and remotely:

`89fb4f1258ac46b41faf5a77e96307e0668d324b045fe618aeb93d9730e663e4`
