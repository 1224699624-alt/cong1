# R177 Boundary-Preserving Separation Arbitrator Result

**Date**: 2026-07-03

**Status**: DONE_NEGATIVE

## Setup

R177 trained a lightweight local MLP arbitrator on original train split using the non-leaking train/val anchor proxy `r100_like_r025b_r097_patch_basic_trainval`.

Selection was performed on original validation only. The selected config was then applied once to locked R110 clean-test-v2 masks.

No new/reannotated test split was used.

## Artifacts

- Training script: `scripts/train_r177_separation_arbitrator.py`
- Launcher: `outputs/bridge_logs/run_r177_boundary_preserving_separation_arbitrator.sh`
- Monitor: `scripts/monitor_r177_boundary_preserving_result.py`
- Clean-test-v2 metrics: `outputs/analysis/r177_boundary_preserving_separation_arbitrator_clean_test_v2_metrics.json`
- Val summary: `outputs/analysis/r177_boundary_preserving_separation_arbitrator_val_summary.json`
- Result summary: `outputs/analysis/r177_boundary_preserving_separation_arbitrator_result_summary.json`
- Log: `outputs/bridge_logs/r177_boundary_preserving_separation_arbitrator.log`

## Validation Selection

Best validation config:

- zone radius: `3`
- threshold: `0.45`
- margin: `0.20`

Validation mean:

- Dice `0.883761`
- Boundary IoU `0.249020`
- component-count error `133.114583`
- false-bridge `0.0`
- sep-band FP `0.516426`

This already signaled a structural failure: the model eliminated false-bridge by over-splitting masks into many fragments.

## Clean-Test-v2 Result

| Metric | R177 | R110 baseline | Delta |
| --- | ---: | ---: | ---: |
| Dice | `0.900540` | `0.917723` | `-0.017183` |
| Boundary IoU | `0.265109` | `0.251896` | `+0.013213` |
| false-bridge | `0.000000` | `0.790123` | `-0.790123` |
| component-count error | `143.901235` | `2.506173` | `+141.395062` |
| sep-band FP | `0.568864` | `0.201358` | `+0.367506` |

Monitor status: `below_best`.

## Interpretation

R177 proved that local risk-zone editing can increase boundary overlap, but the current free add/delete pixel classifier is not anatomically safe. It reduces false-bridge only by fragmenting the mask into many small components, which destroys component-count consistency and lowers Dice well below R110.

This is not a usable method and should not be reported as an improvement.

## Decision

Close the current R177 free-edit MLP branch.

Do not run:

- wider hidden-size sweeps;
- longer epochs;
- lower threshold sweeps;
- more aggressive add/delete zones.

The failure is mechanism-level: unconstrained pixel edits create over-segmentation fragments.

## Next Direction

If continuing this constraint route, the next attempt must be a conservative **delete-only bridge suppressor** or graph-filtered postprocessor:

- allow deletion only in narrow inter-component gap candidates;
- forbid adding foreground;
- forbid edits that create many new connected components;
- require post-edit component-count guardrails on validation;
- select by Dice non-drop first, then component error / false-bridge.

Candidate next run:

**R178 conservative delete-only bridge suppressor**

It should be non-GPU or very low-cost:

1. derive candidate bridge pixels from R110/R100-like foreground inside GT-defined train/val separation bands;
2. learn or score only deletion decisions;
3. after deletion, remove tiny fragments and reject edits that increase component-count error;
4. select on original val;
5. apply once to R110 clean-test-v2 if validation passes.
