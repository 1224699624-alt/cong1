# Experiment Code Review

**Date**: 2026-06-15 22:46  
**Mode**: `[local-only]`  
**Reason**: no delegated reviewer was invoked in this round; performed local executor review before remote launch.

## Reviewed Changes

- `scripts/train_error_refiner.py`
- `scripts/infer_error_refiner.py`
- `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune.sh`
- `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v3.sh`

## Review Checklist

### 1. Method / proposal alignment

- The low-risk next run is intentionally a parameter-tuned variant of `keepbone_cutgap_v2`.
- The structural next candidate `v3` keeps the `keepbone_cutgap_v2` backbone and adds:
  - local AIC-style keep cue
  - extra precision veto
- This matches the repair plan in `NEXT_CANDIDATE_REPAIR_PLAN.md`.

### 2. Hyperparameter exposure

- New launcher `v2_precision_tune` uses existing argparse surface only.
- New launcher `v3` also uses existing argparse surface plus new `target-mode`.
- No hidden constants were introduced in the launcher interface beyond the intended stage2 behavior inside the model branch.

### 3. Logic sanity

- `allstack_anatomy_roi_keepbone_cutgap_refiner_v3` was added to:
  - parser choices
  - anatomy-mode routing
  - dataset target generation path
  - inference mode allowlist
- `prediction_aware` supervision was extended to include `v3`.
- `precision_rescue` is only enabled for `v3`, so `v2` behavior should remain unchanged.

### 4. Reproducibility / outputs

- New launcher names isolate:
  - checkpoint path
  - threshold sweep path
  - output experiment directory
- Existing evaluation remains against dataset ground truth through `scripts/evaluate_masks.py`.

### 5. Syntax / immediate blockers

- Local syntax check passed:

```bash
python -m py_compile scripts/train_error_refiner.py scripts/infer_error_refiner.py
```

## Blocking Issues

- NONE found in local review.

## Non-Blocking Risks

- `v3` changes the stage2 correction path directly, so its behavior still needs empirical validation on `1433/1475/2982/3040`.
- The new `precision_veto` may over-trim if the gate is too strong; this is why `v2_precision_tune` was launched first as the lower-risk branch.

## Launch Decision

- Launch `v2_precision_tune` immediately.
- Keep `v3` ready for the next run once the low-risk branch result is known.
