# R223 Structured Seam Background Route Plan

Date: 2026-07-08

## Goal

Continue the inverted topology route, but replace the failed R222 pixel MLP with
a structured seam/background method that predicts or selects coherent local
background channels between adjacent epiphysis structures.

The target remains:

- keep Dice, IoU, and Recall competitive with R110/ARAA;
- improve bone-seam adhesion, boundary quality, component merge, and component
  count diagnostics;
- avoid over-erosion or漏分;
- use original train/val for development and keep clean-test-v2 locked.

## Evidence So Far

R221 oracle evidence:

- conservative between-instance gap suppression can improve Dice/IoU slightly,
  improve Boundary IoU/F1, reduce gap FP, and improve component-count error on
  original val.

R222 failure:

- pixel MLP learned nearly constant probabilities in seam-eligible regions;
- all low-threshold mask-level sweeps produced zero final edits;
- continuing the same architecture or more epochs is unlikely to solve the
  bottleneck.

R215/R216 evidence:

- background-channel candidates and partial seam actions contain useful signal;
- hard deletion is unsafe;
- gates based on scalar or local crop features were not robust enough.

## Proposed R223 Direction

Use a structured proposal-and-score formulation:

1. Generate connected seam/background proposals from R110 anchor geometry:
   - shallow anchor boundary bands;
   - narrow necks between adjacent high-curvature/low-distance regions;
   - local background-channel opening potential;
   - proposal components rather than individual pixels.
2. Score each proposal with features that preserve spatial coherence:
   - proposal area, width, skeleton length, and compactness;
   - adjacent anchor component/contact geometry;
   - image intensity/gradient contrast across both sides of the seam;
   - foreground preservation risk from anchor distance and local thickness;
   - background-channel connectivity before/after tentative cut.
3. Apply only conservative accepted proposals:
   - max one or few proposals per image;
   - reject if predicted split is too large or too interior;
   - reject if component-count risk is high;
   - prefer partial seam suppression over full deletion.
4. Validate on original val:
   - Dice/IoU/Recall non-degradation;
   - Boundary IoU/F1 improvement;
   - gap-region FP reduction;
   - component merge and component count MAE non-worsening;
   - hard-case visual audit before any clean-test-v2 application.

## First Experiment

`R223-F0` should be a no-clean-test feasibility audit:

- regenerate structured proposals on original val;
- use GT only to label proposal usefulness/risk for diagnostics;
- report oracle upper bound and grouped-CV deployable frontier separately;
- write JSON/CSV only first;
- only write masks if grouped-CV frontier is safer than R216/R218 references.

Promotion condition:

- accepted proposals on at least a meaningful subset of val images;
- mean Dice delta `>= -5e-4`;
- mean IoU delta `>= -8e-4`;
- mean Recall delta `>= -1e-3`;
- Boundary IoU and Boundary F1 improve;
- gap-region FP decreases;
- component count MAE does not worsen.

Stop condition:

- if useful proposals still cannot be separated from risk proposals under
  image-grouped validation, stop this branch and move to a different model
  family or data/protocol route.

