# R313-R315 clean prior refit status

Date: 2026-07-27

## Scope and protocol

- Dataset: `TSRS_RSNA-Epiphysis` only.
- Reviewed selection: 814 train images and 94 original-val images.
- The original raw dataset is unchanged.
- `clean-test-v2` and all test sets remain locked and unused.
- Final comparison uses the R201 original-resolution evaluator.
- Outputs are isolated under R313, R314, and R315 folders.

## Experiment chain

1. R313 retrains the local relation-prior ensemble on the reviewed 814/94 case selection. This is a case-filtering refit, not a pixel-level relabeling experiment.
2. R313 generates a frozen full-image probability map for every selected image.
3. A deterministic R313 visual audit renders original X-ray, original instance label, raw 0-1 prior heatmap, X-ray/heatmap overlay, and prior/label compatibility overlay. It reads stored arrays and does not use image generation.
4. R314 trains paired plain/prior U-Nets with shared initialization and identical selection rules.
5. R315 adapts the mature R202 nnU-Net checkpoint to a two-channel input, initializes the new prior-channel weights to zero, and retrains with the clean-refit prior.
6. R314 and R315 are evaluated on the same 94 original-val images with R201 metrics and paired visualizations.

## R313 prior training definition

Each candidate adjacent structure pair is represented by a 128 x 128 local crop with three spatial channels: local X-ray, Gaussian map for endpoint A, and Gaussian map for endpoint B. The development branch also receives continuous age and sex conditions. The network jointly predicts (1) whether the pair has a valid separating interface and (2) a local separating-interface heatmap. The retained color-instance masks are unchanged copies of the original masks, so their instance identities and boundaries must not be described as corrected ground truth.

The reviewed selection currently yields:

- train: 13,581 positive + 13,581 negative pairs = 27,162 pairs;
- val: 1,759 positive + 1,759 negative pairs = 3,518 pairs.

Two variants are trained with seeds 2581, 2582, and 2583 for four epochs each:

- `image_centers`: image and endpoint geometry only;
- `image_centers_age_sex`: image, endpoint geometry, age, and sex.

At inference, pair classification probability is multiplied by the pair heatmap, mapped back to the original image, aggregated across candidate pairs, and averaged across the three seeds. The resulting frozen map is supplied as the second input channel to U-Net and nnU-Net. It is not a ground-truth seam at inference time.

## Status at 2026-07-27 12:37 Asia/Hong_Kong

- R313 full clean refit is running locally on the RTX 3060 Laptop GPU.
- Pair-manifest construction completed successfully.
- No error was present in the stderr log.
- R314 and R315 are locked behind a coverage gate requiring exactly 814 nonzero train maps and 94 nonzero val maps.

## Closely related recent peer-reviewed work

1. You et al., *Learning With Explicit Shape Priors for Medical Image Segmentation*, IEEE Transactions on Medical Imaging, 2025. It learns global and local explicit shape priors and embeds them into CNN and Transformer U-Net backbones. DOI: 10.1109/TMI.2024.3469214.
2. Gao, *Training Like a Medical Resident: Context-Prior Learning Toward Universal Medical Image Segmentation*, CVPR 2024. Hermes learns task, modality, and anatomical context priors across heterogeneous datasets.
3. Li et al., *Universal Topology Refinement for Medical Image Segmentation with Polynomial Feature Synthesis*, MICCAI 2024. It trains a backbone-independent refiner using synthetic topology-error distributions rather than errors from one upstream model.
4. Zhao et al., *Learning with Explicit Topological Priors for Chest X-ray Rib Segmentation*, MICCAI 2025. It embeds connectivity and interactivity priors across CNN, Transformer, Mamba, and SAM backbones. The public review records a relevant FP/FN trade-off: suppressing interactions can reduce FP while increasing FN.
5. Jeon et al., *Teaching AI the Anatomy Behind the Scan: Addressing Anatomical Flaws in Medical Image Segmentation with Learnable Prior*, ICCV 2025. A learnable canonical anatomical prior is deformed to patient-specific anatomy and used at global and local scales.
6. Tang et al., *Similarity Memory Prior is All You Need for Medical Image Segmentation*, ICCV 2025. It dynamically learns and updates class prototypes in a memory bank rather than freezing one population-average atlas.
7. Rao et al., *Spatial Prior-Guided Boundary and Region-Aware 2D Lesion Segmentation in Neonatal Hypoxic Ischemic Encephalopathy*, MICCAI 2025. It supplies a spatial prior as an additional input channel and combines it with region and boundary losses, directly supporting the broad R313-R315 integration pattern.
8. Su et al., *Mesh-Prompted Anatomy Segmentation*, MIDL 2026. It deforms a canonical SDF/mesh prior to patient anatomy with boundary-based supervision, supporting deformable rather than rigid population priors.

## Planned improvements after the isolated refit comparison

The current experiment deliberately keeps the old prior architecture fixed so that the effect of excluding manually rejected cases can be measured. It does not measure the effect of corrected instance labels. If false separation and Recall loss remain, the next design should be evaluated by ablation rather than silently folded into R313:

1. hard-negative mining for visually close structures that should not be separated;
2. probability calibration and an abstention gate for uncertain pairs;
3. a two-sided support head to protect high-confidence bone interiors;
4. confidence-weighted pair fusion instead of unrestricted pixelwise maximum;
5. scale-normalized adjacency and crop construction;
6. dynamic prototype/memory updates or patient-specific deformable priors;
7. age-stratified sampling and age/sex shuffle ablations to test whether metadata contributes real information.

## Pending artifacts

- `outputs/analysis/r313_clean_relation_prior.json`
- `outputs/priors/r313_clean_relation/manifest.json`
- `outputs/visualizations/r313_clean_relation_prior_heatmaps_val94/manifest.json`
- `outputs/analysis/r314_clean_plain_unet_val94_r201.json`
- `outputs/analysis/r314_clean_refit_prior_unet_val94_r201.json`
- `outputs/analysis/r315_clean_refit_prior_nnunet_val94_r201.json`
- paired R314 and R315 visualization folders
