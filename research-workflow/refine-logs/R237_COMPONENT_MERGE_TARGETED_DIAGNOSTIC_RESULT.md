# R237 Component-Merge-Targeted Diagnostic Result

## ARIS Status
- Stage: original-val candidate diagnostic
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `go_component_count_gap_safe_editor; do_not_claim_merge_counter_solved`

## Purpose
R236 proved safe micro-corrections, but most edits were tiny gap/FP cleanup and
did not move component count. R237 starts the reversed topology route:

- instead of encouraging foreground bone connectivity, encourage/restore
  background connectivity in bone seams;
- explicitly inspect predicted connected components that overlap multiple GT
  instances;
- propose small seam cuts between paired GT instances, then test whether they
  improve boundary/gap/component diagnostics without recall loss.

This uses GT instances for proposal generation on original val only, so it is an
oracle diagnostic and not deployable evidence.

## Outputs
- Script: `scripts/audit_r237_component_merge_targeted_candidates.py`
- Bounded summary: `outputs/analysis/r237_component_merge_targeted_bounded_val_summary.json`
- Bounded candidates: `outputs/analysis/r237_component_merge_targeted_bounded_val_candidates.csv`
- Bounded log: `outputs/bridge_logs/r237_component_merge_targeted_bounded.log`
- Slow exhaustive partial backup:
  - `outputs/analysis/r237_component_merge_targeted_val_partial40_summary.json`
  - `outputs/analysis/r237_component_merge_targeted_val_partial40_candidates.csv`

## Bounded Settings
The exhaustive version was too slow for iteration, so the official R237
diagnostic uses a bounded search:

- `--max-merged-components-per-image 2`
- `--max-pairs-per-component 2`
- `--seam-radii 2,3,4`
- `--action-fracs 0.15,0.25,0.35`
- `--min-cut-area 2`
- no clean-test-v2 access

Runtime on remote original val: about `14 min` for `96` images.

## Main Result
Over original-val R110 anchors:

- candidate rows: `747`
- images with merge-targeted candidates: `62`
- safe non-merge-gain candidate rows: `735`
- images with oracle-safe best candidates: `61`
- over-erosion proxy: `0`
- mean cut GT foreground fraction: `0.000000`
- mean cut GT gap fraction: `1.000000`

Oracle-safe best per image (`61` images):

| Metric | Mean delta |
| --- | ---: |
| Dice | `+0.000189` |
| IoU | `+0.000316` |
| Recall | `+0.000000` |
| Boundary IoU | `+0.001388` |
| Boundary F1 | `+0.001832` |
| gap-region FP rate | `-0.001034` |
| component count MAE | `-0.065574` |
| merged pred components | `+0.000000` |
| background-connected-by-cut | `0.655738` |

## Interpretation
R237 supports the reversed-topology intuition, but with an important nuance:

- Positive: making seam/background pixels connected is safe in this oracle
  setting. It improves Dice/IoU slightly, improves boundary/gap metrics, keeps
  Recall unchanged, and improves component count MAE in oracle-best selection.
- Positive: accepted candidate cuts are pure GT gap/background under this
  diagnostic (`cut_gt_fg_frac=0`, `cut_gt_gap_frac=1`), so the signal is not
  coming from over-erosion or漏分.
- Negative: the strict `merged_pred_components` counter does not improve under
  the current definition. Therefore this route should not be claimed as solving
  the instance-overlap merge counter yet.
- Practical: the exhaustive candidate search is too slow; bounded search is the
  right development loop.

## SATLoss / Topology Note
Topology-preserving segmentation losses such as SATLoss and spatial-aware
persistent feature matching are most naturally aligned with structures where
foreground connectivity should be preserved, such as vessels or roads. For this
project, directly encouraging foreground connectivity is risky because the
failure mode is adjacent bones being connected. The useful adaptation is the
opposite framing: treat bone seams as background channels whose connectivity
should be preserved or restored.

R237 is the first concrete diagnostic of that adaptation.

## Next Step
Build `R238` as a bounded mask-level editor/scorer:

- generate the same bounded R237 candidate family without using GT on target
  selection;
- select candidates using deployable features already available from R236/R237:
  seam probability, local background channel features, cut size, image gradient,
  shallow in-mask distance, and conservative recall/precision proxies;
- target the `safe_nonmerge_gain` profile first, not the strict
  `merged_pred_components` profile;
- compare against R236 on original val:
  - edited coverage should exceed `20/96`;
  - Recall delta should remain `0`;
  - Boundary IoU/F1 and gap FP should improve more than R236;
  - component count MAE should move negative or at least not worsen;
  - visual audit must confirm no over-erosion/leakage.

Do not promote to R201 clean-test-v2 until R238 has a non-oracle mask-level
result on original val.
