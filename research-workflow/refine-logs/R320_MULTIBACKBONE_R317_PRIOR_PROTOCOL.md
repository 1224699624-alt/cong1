# R320 Multi-Backbone R317 Prior Protocol

## Objective

Test whether the frozen image-centers-only prior and the R317 continuous
background penalty improve TSRS_RSNA-Epiphysis segmentation across CNN,
self-configuring CNN, Transformer, Mamba, and prompt-free SAM backbones.

## Locked data policy

- Target dataset: `TSRS_RSNA-Epiphysis` only.
- Train: 875 labeled images.
- Development/model selection: 96-image original validation split.
- `TSRS_RSNA-Articular-Surface` is excluded.
- `clean-test-v2` and the original test split are excluded from coefficient,
  threshold, checkpoint, and model selection.
- Raw datasets are not modified. Every R320 artifact is isolated under
  `outputs/experiments/r320_multibackbone` and `outputs/analysis/r320_*`.

## Frozen prior

- Prior maps: `outputs/priors/r317_image_centers_only/{train,val}`.
- Prior model: the three selected image-centers-only R317 checkpoints.
- Age and sex are not inputs.
- The same frozen prior maps are supplied to every backbone.

## Paired arms

For each backbone, the control and prior arms share the exact same model
initialization and training split.

- Control: a two-channel input interface with an all-zero prior channel and the
  native segmentation loss.
- Prior: the same network receives the frozen R317 prior channel and adds the
  continuous-prior penalty.
- Prior-channel weights are initialized to zero wherever a pretrained model
  requires a new input adapter, preserving the pretrained image-only function.

## Exact prior penalty

For per-sample prior map `p`, target `y`, and foreground-vs-background logit
odds `z`:

```text
p_norm = (p - min(p)) / (max(p) - min(p) + 1e-6)
w = p_norm^2 * 1[y == background]
L_prior = sum(w * softplus(z)) / sum(w)
L_total = L_native + 0.035 * L_prior
```

`alpha=0.035` is locked across all primary backbone experiments. It is not
re-optimized per backbone. Any later per-backbone coefficient sensitivity run
must be labeled diagnostic and cannot replace the locked-coefficient result.

## Backbones

1. Compact U-Net (`InstanceSeparationUNet`).
2. nnU-Net v2, using the existing mature R202 initialization and R317 trainer.
3. Official 2D TransUNet implementation.
4. Official Swin-UMamba implementation.
5. Official MedSAM image encoder/mask decoder in prompt-free mode: no points,
   boxes, or mask prompts at training or inference.

## Training and selection

- First gate: short smoke run for data, logits, gradients, and no-GT-inference.
- Mature runs use paired seeds and a fixed maximum budget with validation early
  stopping only after the minimum epoch.
- The primary checkpoint signal is standard original-val foreground Dice; EMA
  pseudo-Dice is not treated as the final metric.
- Threshold 0.5 is the locked first comparison. Threshold sweeps, if later
  justified, are original-val-only and shared between paired arms.

## Evaluation

R201-style original-val evaluation must include Dice, foreground IoU, Recall,
Boundary IoU/F1, Surface Dice 2px/5px, HD95, ASSD, gap-region FP rate,
component merge rate, and component-count MAE.

The hard gate for each backbone is:

```text
delta Dice > 0
delta foreground IoU > 0
delta HD95 < 0
delta ASSD < 0
```

Broad model-independence requires positive Dice/IoU deltas on all four core
backbones and clear HD95/ASSD improvements on at least three. MedSAM is an
additional foundation-model stress test rather than a prerequisite for the core
claim.

## Launch state

- Server: `connect.westc.seetacloud.com:15655`.
- GPU: one RTX 4090 24 GB.
- Remote workspace: `/root/autodl-tmp/YOLO_SAM_generic_src`.
- Active first mature run: `r320_unet_full_exact`.
- Active log: `outputs/bridge_logs/r320_unet_full_exact.log`.

## 2026-07-29 pause and recovery state

- The exact U-Net pair completed normally; it did not crash.
- The GPU became idle because the queue was waiting for the 461,217,452-byte
  TransUNet checkpoint, whose remote download was unusually slow.
- Two stale `wget` processes were found writing the same TransUNet path. All
  R320 queue/download processes were stopped, and the partial file was kept at
  218,001,716 bytes for a safe resume.
- The MedSAM partial upload was kept at 270,925,824 / 375,049,145 bytes.
- `causal_conv1d` imports successfully; `mamba_ssm` remains to be installed by
  the repaired preparation script.
- `run_r320_prepare_assets.sh` now uses `flock`, exact byte-size validation, and
  `.part` files. The training queue now fails fast on incomplete assets rather
  than appearing to run while waiting indefinitely.
- `scripts/sync_single_file_paramiko.py` now resumes an existing `.part` upload.
- User-requested terminal state: no screen sessions, no R320 process, and zero
  GPU memory use. Do not resume until the user explicitly asks.
