# R200 R110 vs ARAA Constraint Audit

## Purpose

Compare the current best internal system R110 against the trained ARAA baseline using the same clean-test-v2 split and the same bridge/boundary/component metric script.

## Data And Artifacts

- Dataset: `TSRS_RSNA-Epiphysis_clean_test_v2/test`
- Ground truth: `data/raw_variants/TSRS_RSNA-Epiphysis_clean_test_v2/test_labels`
- R110 masks: `outputs/ablations_variants/r110_r100_r108_patch_basic/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks`
- ARAA masks: `outputs/ablations_variants/araa_danet_epoch98/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks`
- ARAA source checkpoint: remote `/home/shenzeyu/workspace/ARAA-Net/ckpt/DANet/98.pth`
- Sync summary: `outputs/analysis/araa_epoch98_sync_summary.json`
- Full audit: `outputs/analysis/r200_r110_vs_araa_bridge_boundary_audit.json`

ARAA epoch 98 is the best checkpoint in the existing ARAA clean-test-v2 sweep, so no new ARAA training was required.

## Results

| System | Dice | IoU | Precision | Recall | Boundary IoU | Boundary F1 | Sep-band FP | False bridge | Component error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R110 current best | `0.917723` | `0.848570` | `0.913343` | `0.923580` | `0.251896` | `0.394721` | `0.201358` | `0.790123` | `2.506173` |
| ARAA DANet epoch98 | `0.911766` | `0.838436` | `0.910177` | `0.914691` | `0.235265` | `0.374350` | `0.201259` | `0.679012` | `2.086420` |
| R110 - ARAA | `+0.005957` | `+0.010134` | `+0.003166` | `+0.008889` | `+0.016631` | `+0.020371` | `+0.000099` | `+0.111111` | `+0.419753` |

Lower is better for `Sep-band FP`, `False bridge`, and `Component error`.

## Interpretation

R110 is stronger than ARAA on the standard overlap and boundary metrics:

- Dice improves by `+0.005957`.
- IoU improves by `+0.010134`.
- Boundary IoU improves by `+0.016631`.
- Boundary F1 improves by `+0.020371`.

However, R110 is worse than ARAA on the anatomy-separation metrics:

- false-bridge rate is higher by `+0.111111`;
- component-count error is higher by `+0.419753`;
- separation-band FP is effectively tied but slightly worse.

Therefore, the current evidence does not yet support the desired claim that R110 solves bone-gap adhesion while preserving Dice/IoU. The supported claim is narrower: R110 improves region and boundary quality over ARAA, but it under-separates components more often.

## Decision

The next model goal should not be generic Dice improvement. It should use R110 as the anchor and target only the ARAA-favorable failure mode:

- keep Dice and IoU at least at R110 level;
- preserve or improve R110 Boundary IoU/F1;
- reduce false-bridge rate below ARAA `0.679012` or at least below R110 `0.790123`;
- reduce component-count error toward or below ARAA `2.086420`;
- do not increase separation-band FP beyond R110/ARAA tied level.

This supports a local bridge-splitting / component-separation correction rather than another full-image source model.
