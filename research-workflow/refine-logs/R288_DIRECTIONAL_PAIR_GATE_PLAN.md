# R288 Directional Confidence-Gated Pair Prior Plan

## Change from R287

R287 applies the same positive pair correction to both incident bone channels. R288 uses a
continuous directional gate for endpoint `i` of pair `(i,j)`:

`gate_i = P_j * 4 * P_i * (1 - P_i)`

Thus the other bone must provide support and the corrected bone must be uncertain. Both-high,
both-low, and already-certain regions receive little or no correction. This is shared across
all pairs; no validation-negative pair is manually removed.

## Fixed protocol

- RAM-W600 train/validation development only; official test remains locked.
- Same mature R285 nnU-Net baseline, 15 training-selected pairs, and radius-7 interface ROI.
- Backbone remains fully frozen.
- Same R287 pair-boundary loss weight `0.01`.
- Validation macro-Dice gate followed by overlap NSD 2 px checkpoint selection.
- Fixed threshold `0.5`; no threshold or pair-list search.
- Isolated output: `outputs/ram_w600/r288_directional_pair_gate_dev`.

## Running status

- Local CUDA process: PID `26348`.
- Epoch 0 passed the macro gate.
- Validation macro Dice: `0.978138208 -> 0.978134453` (`-0.000003755`).
- Validation overlap Dice: `0.872642507 -> 0.872658398` (`+0.000015891`).
- Validation overlap NSD 2 px: `0.878208674 -> 0.878329418` (`+0.000120744`).
- Official test remains unevaluated.

## Final development result

- Status: complete without errors; official test remained locked.
- Selected checkpoint: relation warm-up epoch 4 by validation overlap NSD 2 px.
- Validation macro Dice: `0.978138208 -> 0.978134751` (`-0.000003457`).
- Validation overlap Dice: `0.872642507 -> 0.872817477` (`+0.000174970`).
- Validation overlap NSD 2 px: `0.878208674 -> 0.878516245` (`+0.000307572`).

## R287 comparison and decision

- R287 deltas were macro Dice `-0.000005186`, overlap Dice `+0.000209698`, and overlap
  NSD `+0.000321603`.
- R288 protects macro Dice slightly better and reduces the magnitude of the two largest
  negative-NSD pairs, but it has the same number of negative-NSD pairs (`5/15`), one more
  negative-Dice pair (`3/15` versus `2/15`), and slightly weaker aggregate local gains.
- Decision: do not replace R287 with the current R288 gate. Retain R287 as the stronger
  overlap branch; keep directional gating as a useful ablation and redesign its contextual
  support only when implementing the symmetric gap branch.
