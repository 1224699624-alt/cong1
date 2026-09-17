# R275 Mature Pair Status

R275 is the canonical TSRS original-val nnU-Net baseline pair after training maturity and
protocol alignment. The active checkpoints use the original 512x512 plan; `checkpoint_final.pth`
is retained unpacked and `checkpoint_best.pth` is stored in the compressed archive.

## R275 control (96 original-val images)

- Dice: `0.897965467`
- Recall: `0.899057372`
- Boundary IoU/F1: `0.234335039` / `0.373104700`
- gap-region FP: `0.187305448`
- component merge rate: `0.510416667`
- component count MAE: `2.333333333`
- Surface Dice 2px/5px: `0.650055558` / `0.900421882`
- HD95/ASSD: `16.350589` / `6.165032` px
- Final checkpoint SHA256: `658575e68f94b101fdbc4d861e094bb80882794c5a8762a16d8587d09a5278b9`

## R275 seam variant

The seam-supervision variant is retained as a historical paired comparator with its R201 JSON,
visualizations, final checkpoint, and the compressed best-checkpoint archive. It is not the
canonical baseline for the current local-fusion experiments.

All numbers are from `outputs/analysis/r275_mature_control_original_val_r201.json` and
`r275_mature_seam_original_val_r201.json`; clean-test-v2 was not used.
