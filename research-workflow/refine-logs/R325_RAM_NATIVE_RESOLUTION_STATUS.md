# R325 RAM-W600 native-resolution nnU-Net overlap IEM

## Experiment definition

- Dataset: RAM-W600 only.
- Split: 425 train / 69 validation; test is locked and unused.
- Baseline: native-resolution plain nnU-Net initialized from the same mature R285 checkpoint as R324.
- Improved arm: identical native-resolution nnU-Net plus R324 explicit overlap-IEM loss, lambda 0.002.
- Spatial change: no 384 x 384 resize. Original 600 x 600 and 720 x 720 images are padded to 608 x 608 and 736 x 736 for the network stride.
- Threshold: fixed at 0.5; no threshold search.
- Metrics: Macro DSC/IoU, Overlap DSC/IoU/VOE, Overlap NSD@2px, Overlap MSD.
- Output isolation: `outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem`.
- Seed: 3241.

## Intended comparison

1. R325 native plain versus R324 384 plain: effect of native image pixels.
2. R325 native prior versus R325 native plain: effect of the overlap prior at native resolution.
3. R325 native prior versus R324 lambda 0.002: combined practical result.

## Status

Launched on westd port 27945 at 2026-08-03 16:18 (Asia/Hong_Kong).

- Screen: `r325_native_ram`
- Remote root: `/root/autodl-tmp/YOLO_SAM_generic_src`
- Log: `outputs/bridge_logs/r325_native_ram.log`
- GPU: RTX 4080 SUPER 32 GB
- Largest-image smoke test: 736 x 736 forward/backward completed, peak allocated memory 1.10 GiB.
- First plain-native epoch: 41.82 seconds, Macro DSC 0.971535, Overlap DSC 0.835385, Overlap NSD@2px 0.726896.
- Estimated paired-run duration: approximately 25--50 minutes depending on early stopping.

## Final result

- Completed normally at 2026-08-03 17:02 (Asia/Hong_Kong); both arms ran 30 epochs.
- Selected checkpoint: epoch 23 for both plain-native and prior-native.
- No test data were used.

| Configuration | Macro DSC | Macro IoU | Overlap DSC | Overlap IoU | NSD@2px | MSD px (lower is better) |
|---|---:|---:|---:|---:|---:|---:|
| R324 384 plain | 0.978083372 | 0.957320392 | 0.872430693 | 0.773727131 | 0.879367818 | 1.129427633 |
| R324 384 prior, lambda 0.002 | 0.978095233 | 0.957342386 | 0.872682626 | 0.774123522 | 0.879411574 | 1.129845896 |
| R325 native plain | 0.978762686 | 0.958642006 | 0.872642472 | 0.774060183 | 0.780405084 | 1.968320915 |
| R325 native prior, lambda 0.002 | 0.978736699 | 0.958592892 | 0.872828003 | 0.774352191 | 0.779616470 | 1.931709348 |

### Locked within-native comparison

- Macro DSC: -0.000025988
- Macro IoU: -0.000049114
- Overlap DSC: +0.000185531
- Overlap IoU: +0.000292008
- Overlap NSD@2px: -0.000788614
- Overlap MSD: -0.036611566 px

Verdict: native resolution improves the plain model's global area metrics relative to the 384 configuration, but the overlap prior still creates a trade-off. It improves overlap area and native-pixel MSD while slightly reducing Macro DSC/IoU and NSD, so it does not satisfy the project's hard non-degradation requirement.

Cross-resolution NSD@2px and raw-pixel MSD are not directly comparable: 2 native pixels represent a smaller relative tolerance than 2 pixels after resizing to 384. A scale-normalized surface reevaluation is required before making any R324-versus-R325 boundary claim. Within R325, the plain/prior comparison remains valid because both use identical native spatial handling.
