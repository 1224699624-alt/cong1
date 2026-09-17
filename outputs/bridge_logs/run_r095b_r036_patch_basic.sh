#!/usr/bin/env bash
set -euo pipefail
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES=0
/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python scripts/train_patch_disagreement_arbitrator.py   --anchor-exp r025b_anchor_local_pixel_residual_sam_hqsam   --apply-anchor-exp r038_aggr_r1_t045_m000   --candidate-exps r038_r036_medsam_gbc_adapter_norm   --apply-candidate-exps r038_r036_medsam_gbc_adapter_norm   --output-exp r095b_r036_patch_basic   --checkpoint outputs/patch_arbitrator/r095b_r036_patch_basic.pt   --metrics-json outputs/analysis/r095b_r036_patch_basic_clean_test_v2_metrics.json   --feature-mode basic   --zone-mode anchor_candidate_disagreement   --train-zone-radius 1   --crop-size 80   --crops-per-image 1   --epochs 6   --batch-size 16   --base-channels 16   --lr 0.001   --pos-weight-scale 0.45   --zone-loss-weight 5.0   --boundary-radii 0,1   --prob-thresholds 0.50,0.55,0.60,0.65,0.70   --edit-margins 0.05,0.10,0.15,0.20   --seed 202606951
