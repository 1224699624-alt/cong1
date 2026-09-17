# R261 Experiment Code Review

**Review mode**: local-only (sub-agent delegation was not requested).  
**Scope**: conditional prior builder, 13-channel nnU-Net data path, mature-checkpoint adaptation, relation loss, early stopping, inference, R201 evaluation, and visualization.

## Blocking issues found and fixed

1. `nnUNetv2_plan_experiment` in nnU-Net 2.8.1 does not accept `-c 2d`; removed the invalid argument and retained `-c 2d` only for `nnUNetv2_preprocess`.
2. Validation inference preparation originally enumerated `val_labels`; it now enumerates `filtered_val.csv` and reads no validation-label content.
3. The proposal loader reopened each validation image for every proposal row; added a per-image shape cache to avoid an otherwise severe full-build slowdown.
4. Full-bank scoring originally scanned every bank sample for every candidate pair; replaced it with a standardized `cKDTree` neighborhood query plus continuous age/sex weighting.
5. Newly added nnU-Net channels would lose confidence amplitudes under z-score normalization; the plans patch now keeps X-ray z-score normalization and uses `NoNormalization` for all 12 prior channels.
6. Full-resolution float32 copies of 12 maps would require roughly 80 GB; removed redundant `.npy` writes and retained the exact PNG channels consumed by nnU-Net plus per-case JSON provenance.

## Integrity checks passed

- Source prior artifact is built only from original train instance labels and train metadata.
- Train candidate centers are OOF; original-val candidate centers are frozen image-only proposals.
- Train and validation identities are disjoint.
- Paths containing `clean-test` or `articular` are rejected.
- Six independent seam/support slots are preserved instead of one merged background heatmap.
- The mature X-ray convolution and all segmentation heads are inherited; 12 new input channels are zero-initialized.
- Relation loss uses the original binary mask only as a validity gate and compares seam probability against bone-support probability; it does not create a new hard seam label.
- Confidence abstention produces all-zero slots and therefore recovers the native full-image path.
- Native nnU-Net loss remains unchanged, and the relation coefficient is warm-started and gradient-ratio bounded.

## Remaining sanity requirements before full deployment

- Confirm `NoNormalization` is accepted by the installed nnU-Net 2.8.1 preprocessor.
- Confirm strict 13-channel checkpoint load on the real R202 checkpoint.
- Confirm a GPU batch has finite nonzero `valid_slots`, `relation_loss`, and `alpha`.
- Inspect full-bank train/original-val slot coverage and abstention before allowing the 60-epoch run.

## Non-blocking risks

- The existing proposal generator is not a named anatomical landmark detector; the six slots are confidence-ranked relations rather than fixed metacarpal IDs.
- nnU-Net intensity augmentation may perturb prior amplitudes, although spatial transforms remain aligned and the trainer clamps maps to `[0,1]`.
- The initial best-checkpoint rule remains EMA foreground Dice; R201 seam metrics are evaluated after training and cannot tune clean-test-v2.

**Verdict**: READY FOR REMOTE SANITY ONLY; full training remains gated.
