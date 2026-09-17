# R248-F0 R244 Context Crop Scorer Plan

## ARIS Status
- Stage: implementation-ready original-val scorer
- Dataset: `TSRS_RSNA-Epiphysis`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `ready_to_run_after_r244_fullval`

## Purpose
R245/R247 showed that scalar GT-free filters over R244 candidates are unstable.
R248 keeps the R244 GT-free candidate generator but adds a local context crop
scorer. The scorer can inspect image intensity, anchor mask, predicted component,
candidate cut, anchor-distance context, and local ring/gradient context.

This is intended to distinguish:

- true seam/background cuts that separate adhered bones, and
- risky foreground-eroding cuts that merely create a background channel.

## Added Script
- `scripts/train_r248_r244_context_crop_scorer.py`

The script:

1. regenerates R244-style candidates from image + R110 anchor;
2. computes original-val GT metrics for labels/audit only;
3. builds six-channel local crops for each candidate;
4. trains a tiny CNN with image-grouped CV;
5. searches useful/risk probability thresholds;
6. writes JSON/CSV only.

It does not write masks and does not touch clean-test-v2.

## Smoke Test
Command:

```bash
python scripts/train_r248_r244_context_crop_scorer.py \
  --limit 4 \
  --dist-percentiles 8 \
  --max-candidates-generated 8 \
  --max-components-per-image 2 \
  --epochs 1 \
  --batch-size 16 \
  --grouped-cv-folds 2 \
  --device cpu \
  --output-json outputs/analysis/r248_r244_context_crop_scorer_smoke.json \
  --grid-csv outputs/analysis/r248_r244_context_crop_scorer_smoke_grid.csv \
  --probed-csv outputs/analysis/r248_r244_context_crop_scorer_smoke_probed.csv
```

Smoke result:

- candidate rows: `10`
- candidate images: `2`
- crops: `10`
- safe-useful: `1`
- risk: `9`
- grouped CV: completed
- best: `null` because the smoke is tiny and selects no valid threshold row
- clean-test-v2 used: `false`
- writes masks: `false`

The smoke is a schema/flow check only, not evidence of effectiveness.

## Remote Readiness
The R248 script and launcher have been synced to the remote workspace:

- `/home/shenzeyu/workspace/YOLO_SAM_generic_src/scripts/train_r248_r244_context_crop_scorer.py`
- `/home/shenzeyu/workspace/YOLO_SAM_generic_src/run_r248_r244_context_crop_scorer_fullval.sh`

Remote `py_compile` passed in the `yolo-sam-gpu` environment.

Launcher added locally and remotely:

- `run_r248_r244_context_crop_scorer_fullval.sh`

Do not start the launcher until R244 full-val is complete or explicitly paused,
because R248 regenerates R244 candidates and is also CPU-heavy.

## Next Step
Do not launch R248 full-val while R244 full-val is still CPU-active, because R248
regenerates R244 candidates and would compete for the same CPU-heavy work.

After R244 reaches `96/96` and final files are synced:

1. run R245/R246 final audit on the completed R244 table;
2. if scalar filters remain no-go, launch R248 full original-val grouped-CV;
3. promote to an isolated original-val mask-level editor only if R248 selected
   rows pass:
   - safe rows > risk rows;
   - Recall delta non-negative;
   - Boundary IoU/F1 positive;
   - gap FP negative;
   - mean cut GT foreground fraction <= `0.25`;
   - no clean-test-v2 use.
