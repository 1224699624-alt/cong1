# R231 R230 Seam Visual Audit Result

## ARIS Status
- Stage: original-val visual audit
- Dataset: `TSRS_RSNA-Epiphysis`
- Source masks: `r230_seam_prob_strict_recall_expanded_pilot32`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `R231 = visual_support_for_R230_safety; coverage_claim_not_supported`

## Purpose
R230 produced the cleanest seam-probability mask-level result so far, with two
accepted pure-gap cuts. R231 visually audits those edited cases to check whether
the quantitative gains correspond to plausible seam separation and whether the
edits appear to cause over-erosion or漏分.

## Implemented Artifact
- Script: `scripts/build_r231_r230_seam_hardcase_visualizations.py`

Outputs:
- HTML index: `outputs/analysis/r231_r230_seam_hardcase_visualizations/index.html`
- JSON index: `outputs/analysis/r231_r230_seam_hardcase_visualizations/r231_r230_visual_audit_index.json`
- CSV index: `outputs/analysis/r231_r230_seam_hardcase_visualizations/r231_r230_visual_audit_index.csv`
- Panels:
  - `outputs/analysis/r231_r230_seam_hardcase_visualizations/panels/01_13867_r230_seam_panel.jpg`
  - `outputs/analysis/r231_r230_seam_hardcase_visualizations/panels/02_15067_r230_seam_panel.jpg`

Panel legend:
- GT boundary: green
- prediction boundary: magenta
- TP foreground tint: blue
- FP: red
- FN: yellow
- accepted cut: orange if removed background/gap, red if removed GT foreground

## Cases Audited

### `13867.png`
Quantitative result:
- cut pixels: `8`
- Dice delta: `+0.000073`
- IoU delta: `+0.000114`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000010`
- Boundary F1 delta: `+0.000015`
- gap-region FP delta: `-0.000341`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

Visual audit:
- The accepted cut lies in the local seam/background channel between adjacent
  structures.
- No red foreground-deletion pixels are visible in the accepted-cut panel.
- The R230 mask remains visually aligned with the GT contour; no obvious
  thinning, missing bone, or new leak is visible in the crop.

### `15067.png`
Quantitative result:
- cut pixels: `10`
- Dice delta: `+0.000073`
- IoU delta: `+0.000130`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000270`
- Boundary F1 delta: `+0.000325`
- gap-region FP delta: `-0.000377`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

Visual audit:
- The accepted cut lies along the narrow background seam next to the adjacent
  bone boundary.
- No red foreground-deletion pixels are visible in the accepted-cut panel.
- R230 does not visibly erode the bone body or introduce obvious漏分 in the
  cropped panel.

## Interpretation
The visual audit supports the R230 safety claim for these two edited examples:

- The edits look like seam/background-channel corrections rather than arbitrary
  bone erosion.
- The accepted-cut overlays show removed pixels in background/gap regions, not
  GT foreground.
- The edited masks do not show obvious qualitative漏分 in the inspected crops.
- The visual evidence is consistent with the quantitative findings: Recall
  stays unchanged, component count MAE does not worsen, and gap FP decreases.

## Limitations
This visual audit does not prove general improvement:

- only `2` edited cases are shown;
- the pilot evaluated only the subset with local R110 anchors available;
- no clean-test-v2 images were used;
- no full-val mask-level R201 evaluation has been run;
- coverage remains the main bottleneck.

## Decision
R231 supports keeping R230 strict expanded as the current safe development
checkpoint. The next experiment should improve coverage while preserving the
same visual and metric safety properties.

## Next Step
R232 should attempt probability-ridge candidate generation:

1. Generate candidate cuts directly from high-probability seam ridges.
2. Keep R230 safety gates:
   - zero Recall loss;
   - no component count MAE worsening;
   - no over-erosion proxy;
   - positive boundary/gap-FP deltas.
3. Produce both mask-level original-val metrics and hard-case visual panels.
4. Do not use clean-test-v2 until full original-val behavior is stable.
