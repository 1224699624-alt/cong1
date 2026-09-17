# R226 Background-Connectivity Seam Result

## ARIS Status
- Stage: train/val diagnostic only
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Split: original `val`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `R226-F0 = diagnostic signal only; no mask-level promotion yet`

## Idea
This route follows the reverse-topology hypothesis: instead of encouraging
foreground/bone connectivity, connect nearby background regions through a thin
foreground corridor. If the corridor lies in a true epiphyseal gap, it should
separate adhered bones and reduce gap-region false positives without damaging
the bone mask.

Candidate generation is GT-free. It uses the image and R110 anchor mask only.
GT is used only on train/val to label candidate usefulness/risk and measure
metric deltas.

## Implemented Artifact
- Script: `scripts/audit_r226_bg_connectivity_seam_scorer.py`
- Main pilot outputs:
  - `outputs/analysis/r226_bg_connectivity_seam_r224pilot32_summary.json`
  - `outputs/analysis/r226_bg_connectivity_seam_r224pilot32_candidates.csv`
  - `outputs/analysis/r226_bg_connectivity_seam_r224pilot32_groupedcv.csv`
- Smoke outputs:
  - `outputs/analysis/r226_bg_connectivity_seam_r224smoke3_summary.json`
  - `outputs/analysis/r226_bg_connectivity_seam_r224pilot12_summary.json`

## Pilot32 Result
Pilot command used R224-positive case ordering but did not use GT for candidate
generation:

```bash
python scripts/audit_r226_bg_connectivity_seam_scorer.py \
  --split val \
  --source-candidate-csv outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv \
  --limit 32 \
  --max-components-per-image 1 \
  --max-bg-components 3 \
  --max-pairs-per-component 2 \
  --corridor-radii 1 \
  --action-fracs 0.35 \
  --flush-every 4 \
  --output-csv outputs/analysis/r226_bg_connectivity_seam_r224pilot32_candidates.csv \
  --output-json outputs/analysis/r226_bg_connectivity_seam_r224pilot32_summary.json \
  --cv-csv outputs/analysis/r226_bg_connectivity_seam_r224pilot32_groupedcv.csv
```

Summary:
- Candidate rows: `16`
- Images with candidates: `9`
- Quick useful: `6/16`
- Hard risk: `10/16`
- Safe-gap positive: `3/16`
- Over-erosion proxy: `9/16`

Overall mean deltas:
- Dice: `-0.000041`
- IoU: `-0.000071`
- Recall: `-0.000155`
- Boundary IoU: `-0.000188`
- Boundary F1: `-0.000233`
- Gap-region FP rate: `-0.000191`
- Component count MAE: `+0.000000`

Oracle best per image:
- Rows/images: `6/6`
- Dice: `+0.000014`
- IoU: `+0.000023`
- Recall: `-0.000101`
- Boundary IoU: `+0.000229`
- Boundary F1: `+0.000288`
- Gap-region FP rate: `-0.000354`
- Component count MAE: `+0.000000`

Grouped-CV selector:
- Best accepted rows: `0`
- Promotion gate: failed

## Interpretation
The background-connectivity hypothesis has weak but real candidate-level
signal: some GT-free candidates reduce gap FP and improve boundary metrics
while keeping Dice/IoU nearly unchanged. However, the current geometric
selector is not reliable enough:

- Candidate coverage is low: only `9/32` R224-positive ordered images produced
  any candidates under the fast pilot setting.
- Candidate purity is poor: `10/16` candidates are hard-risk and `9/16` trigger
  the over-erosion proxy.
- The safe-gap subset is too small and does not consistently improve boundary
  metrics.
- Cross-validated useful/risk scoring accepted no candidate, so this should not
  write edited masks or advance to R201 evaluation.

## Decision
Do not promote R226-F0 to mask-level evaluation. Keep it as evidence that
background connectivity is a plausible primitive, but the current seed-pair
geometry is too weak to identify true seams robustly.

## Next Step
Move to R227 with pair localization learned from R224:

1. Use R224 oracle seam masks as positive supervision for where a separative
   corridor should be.
2. Use R226/R225 hard-risk cuts as negatives.
3. Predict seam likelihood from local crops using GT-free inputs only:
   image, anchor mask, distance transform, boundary band, local background
   contacts, and candidate corridor mask.
4. First target is candidate selection on original val, not mask writing:
   nonzero accepted rows, zero or near-zero hard-risk, positive Boundary IoU/F1
   delta, negative gap FP delta, and no material Dice/IoU/Recall regression.
5. Only after candidate-level grouped-CV passes should R227 write isolated
   train/val masks. Clean-test-v2 remains forbidden for tuning.
