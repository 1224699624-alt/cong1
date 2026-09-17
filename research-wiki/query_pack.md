# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Open Gaps
# Gap Map

_Field gaps with stable IDs._

- **G1**: Preserve thin epiphysis positives without causing under-segmentation.
- **G2**: Keep adjacent bone gaps open while improving contour smoothness.
- **G3**: Separate true method gain from annotation-noise gain in evaluation.
- **G4**: Turn repeated bad-case inspection into a more systematic improvement loop.

## Key Papers (40 total)
- [paper:al2024_lumbar_spine_segmentation] Lumbar spine segmentation in MR images: a dataset and a public benchmark
- [paper:al2025_segmentanybone_universal_model] SegmentAnyBone: A universal model that segments any bone at any location on MRI
- [paper:al2025_totalsegmentator_mri_robust] TotalSegmentator MRI: Robust Sequence-independent Segmentation of Multiple Anatomic Structures in MRI
- [paper:al2026_medical_spine_sagittal] Medical Spine Sagittal MRI Dataset for Segmentation and Foraminal Stenosis detection
- [paper:berger2024_topologically_faithful_multiclass] Topologically Faithful Multi-class Segmentation in Medical Images
- [paper:carneiroesteves2024_plugandplay_framework_curvilinear] A plug-and-play framework for curvilinear structure segmentation based on a learned reconnecting regularization
- [paper:du2024_hand_bone_extraction] Hand bone extraction and segmentation based on a convolutional neural network
- [paper:gao2023_coarsetofine_amodal_segmentation] Coarse-to-Fine Amodal Segmentation with Shape Prior
- [paper:gao2024_training_like_medical] Training Like a Medical Resident: Context-Prior Learning Toward Universal Medical Image Segmentation
- [paper:gao2025_show_segment_universal] Show and Segment: Universal Medical Image Segmentation via In-Context Learning
- [paper:grim2025_efficient_connectivitypreserving_instance] Efficient Connectivity-Preserving Instance Segmentation with Supervoxel-Based Loss Function
- [paper:henys2025_bonedat_database_standardized] BoneDat, a database of standardized bone morphology for in silico analyses
## Recent Relationships (21 total)
  idea:current_yolosam_epiphysis_research --addresses_gap--> claim:C2_clean_test_separates_label_noise_from_method_gain
  exp:error_refiner_unet512_b32_local_test --supports--> claim:C1_outperform_internal_refiners
  exp:clean_test_v2_araa_vs_keepbone_cutgap_refiner_v2 --supports--> claim:C2_clean_test_separates_label_noise_from_method_gain
  exp:clean_test_v2_araa_vs_keepbone_cutgap_refiner_v2 --supports--> claim:C3_gap_and_epiphysis_preservation_are_the_main_remaining_bottleneck
  claim:C1_outperform_internal_refiners --tested_by--> exp:clean_test_v2_araa_vs_keepbone_cutgap_refiner_v2
  exp:araa_clean_test_v2_checkpoint_sweep --tested_by--> claim:C1_outperform_internal_refiners
  exp:araa_clean_test_v2_checkpoint_sweep --supports--> claim:C3_gap_and_epiphysis_preservation_are_the_main_remaining_bottleneck
  exp:interaction_refiner_server_line --supports--> claim:C1_outperform_internal_
