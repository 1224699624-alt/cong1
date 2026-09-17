# R064 Path Block Summary

Date: 2026-06-25

Target: clean-test-v2 Dice > `0.9317660066557425` (original ARAA `0.9117660066557425` + `0.02`).

## Current Best Usable Result

| Run | clean-test-v2 Dice | Precision | Recall | Boundary IoU | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| R038 `r038_aggr_r1_t045_m000` | `0.914357` | `0.899054` | `0.931483` | `0.0` in fast oracle record / `~0.243` in evaluated refiner line | current best usable |

Remaining gap to target: about `+0.017409` Dice.

## Recent Negative Results

| Run | Purpose | clean-test-v2 Dice | Key conclusion |
| --- | --- | ---: | --- |
| R055 `sam_structured_boundary_last2` | SAM structured prompt + boundary losses + last-2 image encoder tuning | `0.770978` | high-recall/low-precision collapse |
| R057 `medsam_structured_boundary_last2` | MedSAM structured counterpart | `0.816591` | same low-precision standalone failure mode |
| R060 `structured_valtrained_anchor_residual_r038_apply` | learn deployable residual from R055/R057 oracle signal | `0.914337` | tied/slightly below R038; structured branch not learnably useful |
| R061 top-diverse historical oracle | test broad historical candidate complementarity | best non-leaking vote about `0.914151`; per-image oracle `0.916817` | selection/fusion cannot reach target |
| R062 old checkpoint clean-test-v2 reevaluation | test old recall/clDice checkpoints | `0.906995` / `0.907401` | old checkpoints are not useful restart points |
| R063 `r025b_keepbone_v2_recall_rescue` | strong-anchor recall-rescue refiner | `0.909564` | below R038 and far below target |
| R064 R038/R025b/R063 oracle | non-GPU complementarity check | pixel oracle `0.925748` | even GT-leaking oracle is below target |

## Interpretation

The R064 diagnostic is the strongest stop signal for this local branch. If R038/R025b/R063 cannot reach target even under pixel oracle, then another learned selector, residual editor, threshold sweep, or candidate fusion over these masks cannot reach the target either.

Together with earlier R049/R052/R058/R061 results, the pattern is stable:

- Whole-image or per-image selection saturates around `0.914-0.917`, far below `0.931766`.
- Pixel oracle sometimes exceeds target when weak high-recall candidates are included, but learned gates repeatedly fail to recover the signal without precision collapse.
- New SAM/MedSAM structured branches add high-recall foreground noise rather than deployable boundary-quality evidence.
- Filtered/reannotated data transfer and old checkpoints do not improve clean-test-v2.

## Path Status

Closed as low-value:

- same-family mask fusion / voting / selector reruns;
- R038/R025b/R063 residual or stack-fusion follow-up;
- structured SAM/MedSAM branch exploitation from R055/R057;
- old checkpoint revival;
- direct filtered/reannotated transfer.

Not enough evidence to call the whole research objective impossible, but the current experiment family is blocked for the revised +0.02 target.

## Recommended ARIS Pivot

Do not launch R065 as another mask-space or same-refiner variant. The next useful ARIS pass should start a genuinely new direction, preferably one of:

1. Data/label audit: identify clean-test-v2 hard-case label characteristics and train/val mismatch before more GPU training.
2. Anatomy-layout model: add explicit metacarpal/epiphysis instance layout constraints instead of pure pixel mask editing.
3. Literature-driven architecture replacement: compact segmentation backbone or region graph model trained end-to-end on epiphysis, using YOLO/SAM only as proposal priors.
4. Test-time or domain adaptation only if it uses image/anatomy consistency, not GT-leaking mask oracle selection.

Minimum next artifact before GPU launch: a new experiment plan that names the new mechanism, expected failure mode it addresses, train/val/test split usage, and why it is not another candidate-mask fusion route.
