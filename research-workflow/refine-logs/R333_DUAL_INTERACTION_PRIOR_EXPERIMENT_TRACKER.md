# R333 Dual-Interaction Prior Experiment Tracker

| Run | Dataset | Purpose | Split | Status |
|---|---|---|---|---|
| R333-S0 | RAM + TSRS | CUDA/state-mask smoke | train/val subset | TODO |
| R333-R | RAM-W600 | seam-input + overlap-output combined loss | train/validation | TODO |
| R333-T | TSRS Epiphysis | seam-input + conservative completion combined loss | train/original-val | TODO |
| R333-A | both | state-gating ablation | validation | BLOCKED on main runs |
| R333-F | both | locked final test | test/clean-test-v2 | BLOCKED on validation gates |

## R334 repair launch (2026-08-12)

| Run | Dataset | Purpose | Split | Status |
|---|---|---|---|---|
| R334-S0 | RAM-W600 | frozen-R332 seam-adapter CUDA/state smoke | 2 train cases | PASS: finite loss, 6,122 trainable parameters |
| R334-R | RAM-W600 | preserve R332 step2 while adding seam separation | train/validation | RUNNING on `connect.westd.seetacloud.com:32478`, screen `r334_ram_frozen` |
| R334-T | TSRS Epiphysis | exact R317 native-space integration repair | train/original-val | PENDING R334-R validation audit; clean-test-v2 locked |

R334-R freezes the mature nnU-Net and R332 iterative refiner. Only the tiny seam adapter is trainable. Its validation baseline is the actual frozen R332 step-2 output, and the gate requires non-degradation of overall DSC/IoU and R332 overlap DSC/NSD/MSD together with non-worse seam FP. RAM test is not opened.

## R335 loss-only conditional-state experiment (2026-08-13)

| Run | Dataset | Purpose | Split | Status |
|---|---|---|---|---|
| R335-S0 | RAM-W600 | state-mask/conditional-loss CUDA smoke | 2 train cases | PASS: finite four-term loss; no new inference module |
| R335-R | RAM-W600 | loss-only fine-tuning of existing R332 refiner | train/validation | RUNNING on `connect.westb.seetacloud.com:30043`, screen `r335_conditional_loss` |
| R335-T | TSRS Epiphysis | unchanged R317 nnU-Net with conditional seam/support loss | train/original-val | QUEUED after R335-R in the same guarded launcher |

R335 uses `Lseg + lambda_sep Msep Lsep + lambda_overlap Moverlap Loverlap + lambda_preserve Lpreserve`. RAM obtains `Moverlap` only from its genuine 14-channel multi-label masks. TSRS sets the untrustworthy overlap state to zero and does not interpret color collisions as projection overlap. Neither RAM test nor TSRS clean-test-v2 is opened. Storage is limited to one best checkpoint, compact histories/results and final validation masks.
# R336 state-PCGrad-selective-distillation follow-up (2026-08-13)

- Motivation: R335 fixed-weight conditional losses improved TSRS Dice/IoU but worsened merge/HD95/ASSD, and degraded RAM overlap metrics.
- Method: mutually exclusive state masks; logit-space projection of conflicting seam/overlap auxiliary gradients against the primary segmentation gradient; selective distillation only on teacher-correct, high-confidence, non-interaction pixels.
- Baselines: RAM R332 step-2; TSRS R317 checkpoint_best.
- Selection: multi-metric validation gate (overall DSC/IoU, overlap DSC/NSD/MSD, seam FP); TSRS final audit uses original-val R201. RAM test and TSRS clean-test-v2 remain locked.
- Artifacts: R335 essential files copied to `outputs/artifact_bundles/r335_conditional_state_loss/` with SHA-256 manifest (13 files, 474,927,292 bytes).
- Storage: removed only remote `outputs/cache` (2.0 GB, rebuildable); remote free space increased from 3.1 GB to 5.0 GB. R317 preprocessing retained for R336.
- Smoke test: passed finite forward/backward/evaluation; gradient projection unit test converts an opposing auxiliary gradient to a non-conflicting one.
- Remote run: screen `r336_state_pcgrad_distill`, server port 30043, started 2026-08-13 14:26 +08:00. Serial order RAM validation then TSRS original-val.
