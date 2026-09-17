# R242 Candidate Crop Scorer Progress

## ARIS Status
- Stage: original-val candidate-level scorer diagnostic
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: no
- Current status: remote full original-val grouped-CV running

## Purpose
R240 showed that GT-free background-channel geometry can move boundary/gap
diagnostics, but often by cutting true foreground. R241 then showed that scalar
candidate features cannot reliably separate:

- true bone-seam/background cuts, and
- risky shallow foreground cuts.

R242 starts the next route: score each candidate from a local crop rather than
from scalar features alone.

## Method
Added:

- `scripts/train_r242_candidate_crop_scorer.py`

The script reuses R239 candidate generation and trains a small CNN with local
crop channels:

- grayscale image crop
- R110 anchor mask crop
- candidate cut mask crop
- anchor interior distance transform
- background distance transform
- R228-style seam-probability crop

Labels use original-val GT only for supervision/audit:

- safe useful:
  - non-worse Dice/IoU/Recall
  - non-worse Boundary IoU/F1
  - gap-region FP improves
  - component count MAE non-worse
  - `cut_gt_fg_frac <= 0.25`
  - `cut_gt_gap_frac >= 0.75`
- risk:
  - foreground-heavy cut, recall loss, overlap drop, or boundary drop

The grouped-CV split is by image. The script writes JSON/CSV only and does not
write masks.

## Smoke
Local CPU smoke completed:

- output: `outputs/analysis/r242_candidate_crop_scorer_smoke.json`
- rows: `6`
- images: `3`
- result: selected `1` risk cut

This smoke is only a schema/flow check. It is not evidence for promotion.

## Remote Run
Launched on the remote server:

- screen: `r242_candidate_crop_fullval`
- Python: `/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python`
- GPU binding: `CUDA_VISIBLE_DEVICES=1`
- log: `outputs/bridge_logs/r242_candidate_crop_scorer_fullval.log`

Target outputs:

- `outputs/analysis/r242_candidate_crop_scorer_fullval_groupedcv5.json`
- `outputs/analysis/r242_candidate_crop_scorer_fullval_threshold_grid.csv`
- `outputs/analysis/r242_candidate_crop_scorer_fullval_probed_rows.csv`

Latest monitor:

- R228/R242 seam-probability sampling finished: `24/24`
- R239-style candidate generation entered original-val loop
- observed progress: `19/96`
- no final JSON/CSV yet
- process is CPU-active, not hung; GPU1 is idle because this phase is
  candidate generation and metric labeling, not CNN training

## Promotion Gate
R242 can only move to an isolated original-val mask editor if grouped-CV shows:

- selected risk far below selected safe-useful rows
- Recall delta stays at `0`
- Boundary IoU/F1 improve
- gap-region FP decreases
- component count MAE does not worsen
- selected rows have low `cut_gt_fg_frac` and high `cut_gt_gap_frac`

Even if this passes, the next step is still original-val mask-level evaluation,
not clean-test-v2.

## Decision Pending
No decision yet. Wait for the remote grouped-CV result.

If R242 fails like R218/R241, the likely conclusion is that this R239 candidate
distribution is too noisy. The next route should be cleaner candidate
generation closer to the R237/R238 component-merge oracle structure, rather
than another threshold sweep over the same candidates.

## Fallback / Bounded Diagnostic
The running full-val job should be allowed to continue. If it crashes or remains
too slow to be useful, use a bounded diagnostic with progress snapshots rather
than restarting another opaque full run:

```bash
cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
CUDA_VISIBLE_DEVICES=1 /home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python \
  scripts/train_r242_candidate_crop_scorer.py \
  --source-mode all-anchors \
  --limit-source 32 \
  --limit-train-rows 300 \
  --samples-per-image 1024 \
  --positive-oversample 512 \
  --max-components-per-image 3 \
  --max-bg-components 6 \
  --max-pairs-per-component 4 \
  --corridor-radii 1,2,3 \
  --action-fracs 0.15,0.25,0.35 \
  --max-candidates-per-image 120 \
  --grouped-cv-folds 4 \
  --epochs 6 \
  --batch-size 64 \
  --device cuda \
  --flush-every 4 \
  --candidate-snapshot-csv outputs/analysis/r242_candidate_crop_scorer_bounded32_candidate_snapshot.csv \
  --progress-json outputs/analysis/r242_candidate_crop_scorer_bounded32_progress.json \
  --output-json outputs/analysis/r242_candidate_crop_scorer_bounded32_groupedcv4.json \
  --grid-csv outputs/analysis/r242_candidate_crop_scorer_bounded32_threshold_grid.csv \
  --probed-csv outputs/analysis/r242_candidate_crop_scorer_bounded32_probed_rows.csv
```

This bounded run is only for directional triage. It cannot replace the full
original-val grouped-CV result for promotion.
