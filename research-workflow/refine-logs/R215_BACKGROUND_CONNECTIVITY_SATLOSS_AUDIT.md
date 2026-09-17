# R215 Background-Connectivity SATLoss Audit

Date: 2026-07-07

## Source

Repository inspected in isolated local directory:

- `.tmp/SATLoss`
- Upstream: `https://github.com/JRC-VPLab/SATLoss.git`

This is a much more useful source than the previous notebook-only repository.
It contains reusable implementation files:

- `utils/losses.py`
- `utils/PDMatching.py`
- `utils/cubical_complex.py`
- `utils/metrics.py`

README points to ICCV workshop paper:

- *Topology-Preserving Image Segmentation with Spatial-Aware Persistent Feature
  Matching*

## Core Implementation

The main loss is `PDMatchingLoss` in `utils/losses.py`.

Important mechanics:

1. `PDMatchingLoss` builds cubical-complex persistence diagrams with
   `CubicalComplex(dim=2)`.
2. It compares prediction and GT persistent diagrams using
   `SpatialAware_WassersteinDistance`.
3. The spatial-aware distance uses both:
   - persistence diagram coordinates;
   - creator pixel coordinates from `PersistenceInformation.pairing`.
4. It computes both:
   - 0-dimensional features: connected components;
   - 1-dimensional features: loops/holes.
5. It pads image boundary by 1 and pads non-square images to square because the
   persistence implementation expects this.
6. Training loop computes:
   - pixel BCE;
   - SAT topology loss on CPU;
   - total loss = BCE + `tloss_w * topology_loss`.

The relevant code path:

- `train.py`: `loss_topo = criterion['topo'](pred.cpu(), target.cpu(), img_names)`
- `utils/losses.py`: `PDMatchingLoss.forward`
- `utils/PDMatching.py`: `SpatialAware_WassersteinDistance.forward`

## Key Insight For Epiphysis

The user's proposed inversion is valid:

> If foreground topology encourages bones to connect, use background/gap
> topology to encourage the bone-seam background to remain connected.

For our task, the topology target should not be "epiphysis foreground is
connected." The target should be:

- adjacent bones remain separated;
- bone-gap/background channels stay open;
- false foreground bridges across seams are penalized;
- background connectivity is preserved in local gap-risk regions.

This is more aligned with the objective than clDice on foreground masks.

## How To Adapt SATLoss Safely

### Do not apply global background topology to the whole image

Global background in X-rays includes huge exterior regions. A whole-image
background SATLoss would mostly learn trivial outside-background connectivity
and could ignore the small seam regions we care about.

### Use local gap-risk crops

Compute topology on local regions around adjacent-bone / bridge-risk zones:

1. start from R110 anchor or candidate mask;
2. identify gap-risk bands:
   - near foreground;
   - between adjacent predicted components;
   - around candidate cuts from R210/R211/R214;
3. crop local windows around these gap-risk bands;
4. define background probability:
   - `bg_prob = 1 - fg_prob`;
   - `bg_target = 1 - fg_gt` for train/val only;
5. run SATLoss or a simpler Betti/connectivity diagnostic on the crop.

### Use it first as a candidate diagnostic, not as full training loss

Lowest-risk R215:

- no mask writing;
- no clean-test-v2 use;
- compute background topology features for train/val candidates;
- add features to candidate CSV;
- test whether grouped image validation improves useful/risk separation.

Potential candidate features:

- local background beta0 before/after cut;
- local background beta0 error vs GT;
- local background component containing the seam area;
- spatial-aware Wasserstein distance between predicted and GT background PDs;
- number/persistence of unmatched foreground bridge features;
- creator coordinate distance to candidate cut center;
- PH loss delta if candidate cut is applied.

### Consider training loss only after candidate evidence

If R215 diagnostics help candidate selection, then R216 can test a training
loss:

`loss = BCE/Dice boundary loss + lambda_bg_topo * SATLoss(bg_prob_crop, bg_gt_crop)`

Strict rules:

- original train/val only;
- small crops only;
- small weight;
- validation gate must check Dice/IoU, Boundary IoU/F1, Surface Dice, HD95,
  ASSD, gap FP, component merge, component count MAE;
- hard-case visual audit must reject over-erosion and missed bones.

## Why This Is Better Than Foreground clDice

Foreground clDice/SATLoss pushes connected foreground structures to remain
connected. That is good for roads/vessels, but dangerous for epiphysis masks.

Background/gap SATLoss instead says:

- the seam/background between neighboring bones should stay open;
- foreground bridges that close the seam damage background topology;
- candidate cuts are good only when they restore seam background without
  damaging bone regions.

This directly targets bone-gap adhesion.

## Implementation Risks

- Local environment currently lacks `gudhi`, `ripser`, and `persim`.
- SATLoss also requires `torch-topological` and `POT`.
- PH computation is CPU-side and may be slow.
- Whole-image PH is likely too noisy for our task.
- Background topology needs carefully designed local crops; otherwise it can
  reward trivial exterior background rather than bone seams.
- GT-derived background topology is train/val-only and must never be used for
  clean-test-v2 tuning.

## Recommended Next Experiments

### R215-A: Dependency and API Smoke

Check remote environment for:

- `gudhi`
- `torch_topological`
- `ot` / POT

If missing, install only in an isolated conda env or keep R215 as optional
offline analysis.

### R215-B: Local Background Topology Diagnostic

Create a no-training script:

- `scripts/audit_r215_background_connectivity_candidates.py`

Inputs:

- candidate CSV/masks from R212/R214;
- R110 anchor masks;
- original train/val labels.

Outputs:

- candidate CSV with background topology features;
- grouped image validation report;
- no masks written.

### R216: Guarded Background-Topology Loss

Only if R215-B improves candidate selection:

- train a crop-level or refiner-level background-topology loss on train/val;
- no clean-test-v2 use until full original-val R201 and visual audit pass.

## Bottom Line

Yes, the user's inversion is the right way to adapt this idea. The promising
direction is not "make bone foreground topology connected"; it is:

**make local bone-gap/background topology connected enough to keep adjacent
bones separated.**

Use SATLoss first as local background-connectivity diagnostics and candidate
features. Promote it to training loss only after it proves useful under
grouped train/val validation.
