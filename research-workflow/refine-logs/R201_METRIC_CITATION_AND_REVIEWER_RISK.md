# R201 Metric Citation And Reviewer-Risk Note

## Standard Metrics

- Dice, IoU/Jaccard, Precision, and Recall are conventional segmentation/classification overlap metrics and are acceptable for the main table.
- Boundary IoU is a boundary-focused segmentation metric from Cheng et al., CVPR 2021.
- Surface Dice was introduced for clinically oriented medical image segmentation evaluation by Nikolov et al. and is appropriate for boundary/surface tolerance claims.
- HD95 and ASSD are common medical image segmentation surface-distance metrics; report them as pixel distances unless reliable physical spacing is available.
- Metrics Reloaded recommends problem-aware metric choice in biomedical image analysis, supporting the use of overlap plus boundary/surface plus task-specific diagnostics.

## Task-Specific Diagnostics

- `gap_region_fp_rate`, `component_merge_rate`, and `component_count_mae` must be described as failure-mode diagnostics for bone-gap adhesion and anatomical consistency.
- Do not present these diagnostics as universal benchmark metrics.
- In the paper, define the exact formula, kernel radius, foreground rule, and low/high direction next to the table or in the appendix.

## Verified Sources

- Boundary IoU: Cheng et al., "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation", CVPR 2021, https://arxiv.org/abs/2103.16562
- Surface Dice: Nikolov et al., "Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy", https://arxiv.org/abs/1809.04430
- Metrics Reloaded: Maier-Hein et al., "Metrics reloaded: recommendations for image analysis validation", Nature Methods, https://www.nature.com/articles/s41592-023-02151-z
- nnU-Net baseline: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods, https://www.nature.com/articles/s41592-020-01008-z
