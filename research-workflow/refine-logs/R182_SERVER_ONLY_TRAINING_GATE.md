# R182 Server-Only Training Gate

Date: 2026-07-03

## User Constraint

Training must not be run locally. Local actions are limited to planning, code/log inspection, launcher preparation, and artifact synchronization. Any GPU training or long inference/evaluation job must run on the server:

- Host: `10.1.115.157`
- User: `shenzeyu`
- Remote workspace: `/home/shenzeyu/workspace/YOLO_SAM_generic_src`
- Python: `/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python`

## Remote Check

Remote state checked through Paramiko.

- GPU0: RTX 4090, idle
- GPU1: RTX 4090, idle
- Active screen sessions: none

The server is available for the next valid training run.

## Current Gate

The R168/R170/R171 reviewed-protocol route is not ready:

- `outputs/analysis/r168_reannotation_protocol_review_package/r168_reannotation_protocol_review_reviewed.csv` is absent locally and remotely.
- `data/raw_variants/TSRS_RSNA-Epiphysis_reannotation_protocol_reviewed_v1` is absent remotely.
- Therefore R171 must not be launched yet.

Launching R171 without the reviewed CSV would bypass the train/val label-protocol gate and would not produce a defensible experiment.

## Decision

No local training was run.

No server training was launched because the next guarded reviewed-protocol route is missing its required human-reviewed CSV and isolated variant.

## Next Valid Actions

1. If the R168 reviewed CSV is exported, run the R172 wrapper, create the R170 isolated variant, then launch R171 on the server only.
2. If the R168 CSV is not available, open a genuinely new model-family/environment route with a strict server-side smoke/overfit gate before any clean-test-v2 diagnostic.
3. Do not continue DINOv3 loss-weight sweeps, old SAM/HQ-SAM/MedSAM fusion, or local training.
