# R085 Gradient Boundary Rule Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R085 tested whether cheap image-gradient evidence can provide a deployable boundary correction signal around the strong R038 anchor. This was a low-cost replacement for the slower R084 GrabCut grid.

## Setup

- Tune anchor: R025b on original val.
- Apply anchor: R038 on clean-test-v2.
- Rule family: edit only a radius-1 boundary band using Sobel gradient magnitude.
- Tiny diagnostic grid:
  - low-gradient removal threshold `0.25`;
  - high-gradient addition threshold `0.70`;
  - max remove fraction `{0, 0.001}`;
  - max add fraction `{0, 0.001}`.

## Result

Validation selected the zero-edit rule:

| Split/System | Dice | Precision | Recall | Boundary IoU | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| original val selected rule | `0.894701` | `0.888486` | `0.906338` | `0.219753` | zero edits, reproduces R025b val |
| clean-test-v2 selected rule | `0.914357` | `0.899054` | `0.931483` | `0.243176` | zero edits, reproduces R038 |

## Interpretation

Simple hand-written gradient thresholds do not provide a useful correction signal. The safest rule is to leave the anchor untouched, which reproduces the current R038 plateau.

This is consistent with R075 and R084: naive image evidence around the mask boundary either runs too slowly when optimized by GrabCut, or collapses to no-op when made cheap and conservative.

## Decision

Do not continue hand-written gradient/Canny/GrabCut boundary rules as a target-seeking path.

Next useful direction should introduce learned or ensemble uncertainty that can rank local edits better than raw image gradients, or pivot to a different literature-driven contour architecture.

