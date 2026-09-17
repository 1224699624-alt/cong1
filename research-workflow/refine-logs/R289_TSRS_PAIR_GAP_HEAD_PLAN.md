# R289 TSRS Pair-Specific Gap Relation-Head Gate

- Dataset: `TSRS_RSNA-Epiphysis` only, 875 train / 96 original-val.
- Baseline feature extractor: exact mature R275 nnU-Net checkpoint, fully frozen.
- Pair graph: top 20 stable instance-ID pairs selected from train-label adjacency only.
- Relation outputs per pair: `gap / overlap / uncertain-support`.
- Supervision: only a radius-8 local interface between the two instance masks; no complete
  inverse-background target is used.
- Inference input: X-ray only. Instance GT is used only to construct train/validation targets.
- Output masks are not edited in R289, so R201 is intentionally deferred to R290.
- `clean-test-v2` and `TSRS_RSNA-Articular-Surface` are locked out.
- Gate: original-val gap Dice >= 0.30, recall >= 0.50, precision >= 0.20.
- Runtime design: four epochs, 800 randomly sampled train pair-crops per epoch; every
  validation evaluates all 369 original-val pair-crops.
- Passing decision: permit R290 frozen-backbone gap residual and full original-val R201 audit.

## Running status

- Local CUDA process started successfully; PID `25352`.
- Pair audit selected 20 train-only stable anatomical instance-ID pairs.
- Expanded training pool: 12,847 pair-crops; sampled 800 per epoch.
- Original-val evaluation pool: all 369 selected-pair crops across the 96 cases.
- GPU memory is about 2.4/6 GB with no runtime error.

## Restart audit

- The first R289 process completed its first train epoch but stopped before validation
  because zero-byte `.bmp/.jpeg/.png` placeholders coexist with the real `.jpg` in
  original-val and the loader treated them as duplicate images.
- No checkpoint or result from that interrupted process is valid.
- The loader now follows deterministic non-empty extension priority and resolves case
  `10520` to its real `.jpg` rather than a placeholder.
- Valid isolated rerun: R289b, PID `9960`, output
  `outputs/tsrs/r289b_pair_gap_head_dev`; training is running on CUDA.

## Final R289b audit under paper-aligned metrics

- Training completed without runtime errors; legacy pooled diagnostics were gap Dice `0.840956`,
  recall `0.981069`, and precision `0.735862` on a validation subset truncated to the first four
  pairs per case. These are internal diagnostics only.
- Complete validation contains `1,376` pair-crops from `95/96` original-val cases; `1,026`
  crops contain a non-empty GT background gap.
- A GT-ROI-clipped oracle evaluation gave DSC `0.819821` and NSD@2px `0.791538`, but this is
  invalid for paper-facing use because GT was used to clip the prediction.
- Correct full-512x512-crop evaluation with MONAI 1.4.0:
  - DSC: `0.008229`.
  - NSD@2px: `0.003539`.
  - VOE: `0.995862`.
  - symmetric MSD: `191.234026 px`.
  - RAVD: `1096.640632`.
  - gap-presence balanced accuracy: `0.5`.
  - specificity: `0.0`; confusion matrix `[TN=0, FP=350, FN=0, TP=1026]`.
- Cause: R289b supervised the relation class only inside the GT interface ROI, leaving the
  outside-crop gap logits unconstrained. The head therefore predicts gap somewhere in every crop.
- Decision: **no-go**. Do not enter R290 and do not use the legacy/oracle gate as evidence.
  The next relation-head version must supervise outside-ROI pixels as non-gap/uncertain or use
  a deployable GT-free predicted interface gate, then be selected by the paper-aligned metrics.
