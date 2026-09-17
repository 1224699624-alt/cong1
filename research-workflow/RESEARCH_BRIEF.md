# Research Brief

## Problem Statement

Improve the current `YOLO+SAM` epiphysis segmentation line until it clearly matches or exceeds the ARAA benchmark under the user's aligned environment, with `clean-test-v2` as the primary decision slice and the original test split kept as a secondary control slice.

## Current Project Position

- The project is no longer in an early exploration phase.
- A strong internal refiner line already exists: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`.
- A strong external anchor also exists: ARAA best `clean-test-v2` checkpoint `98.pth`.
- The remaining gap is small enough that future work must be more disciplined and anatomy-targeted, not just another generic refinement variant.

## Locked Benchmark Context

### External target

- **Method**: ARAA
- **Repo on server**: `/home/shenzeyu/workspace/ARAA-Net`
- **Primary score to beat**: Dice `0.911766` on `TSRS_RSNA-Epiphysis_clean_test_v2`
- **Key profile**: slightly stronger recall and boundary completeness than the current best refiner

### Current internal anchor

- **Method**: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`
- **Primary comparison slice**: `TSRS_RSNA-Epiphysis_clean_test_v2`
- **Recorded score**: Dice `0.910029`
- **Current interpretation**: very close to ARAA, but still slightly too conservative

## Research Goal

Push the current refiner family from "nearly tied with ARAA" to "convincingly competitive or better than ARAA", especially on the cases that currently determine failure:

- thin epiphysis under-segmentation
- narrow inter-bone gap collapse
- overly smooth masks that look cleaner but lose anatomy

## Main Bottleneck

The main remaining difficulty is no longer coarse localization. It is the trade-off between:

- preserving thin positive bone structures
- keeping adjacent bone gaps open
- improving boundary quality without erasing small anatomy

This is why `2982`, `1475`, `3040`, and `1433` remain important qualitative gate cases.

## Evaluation Policy

### Primary slice

- `TSRS_RSNA-Epiphysis_clean_test_v2`
- use this to judge whether a method really beats the current ARAA target under cleaner evaluation conditions

### Secondary slice

- original `TSRS_RSNA-Epiphysis` test
- use this to verify that gains are not only artifacts of evaluation cleanup

### Reporting rule

Every promoted result should report both slices when possible. A result should not be called the new mainline if it only looks good on the cleaned slice while collapsing on the original split.

## Method Direction

Future variants should stay grounded in the current structure-aware refiner line and emphasize:

- thin-structure preservation
- explicit gap protection between neighboring bones
- prediction-aware or structure-aware boundary correction

Methods that merely smooth contours or rely on blunt post-processing are not the main direction unless they also preserve anatomy on hard cases.

## Dataset and Experiment Discipline

- Preserve the original dataset layout.
- Keep `TSRS_RSNA-Epiphysis` and `TSRS_RSNA-Articular-Surface` fully separated.
- Any filtering or suspicious-label removal must create a new manifest and a new isolated dataset variant.
- Filtered-data gains must be reported separately from pure architecture gains.
- Every new experiment must use isolated output folders and explicit launcher names.

## Success Criteria

A new method direction is worth promoting only if it satisfies most or all of the following:

1. Beats the current strongest internal refiner on `clean-test-v2`.
2. Reaches or exceeds the ARAA target Dice `0.911766`, or comes with a very strong qualitative and boundary-quality case if the score is only marginally below.
3. Does not visibly worsen the hard cases centered on thin epiphysis and bone-gap preservation.
4. Keeps the claim chain honest across `clean-test-v2`, original test, and any filtered-data control runs.

## Immediate Next Use In ARIS

This brief is now aligned to support:

- `/experiment-bridge` for the next ARAA-targeted experiment campaign
- structured bad-case-driven method selection
- future `/auto-review-loop` runs that judge whether a candidate should become the new project mainline

## Non-Goals

- Replacing the whole project with a different framework
- Mixing cleaned-data and default-data conclusions into one result line
- Promoting visually smoother masks that actually damage thin anatomy
- Treating small metric movements as meaningful without bad-case validation
