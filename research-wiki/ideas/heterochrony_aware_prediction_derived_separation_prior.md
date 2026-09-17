---
type: idea
node_id: idea:heterochrony_aware_prediction_derived_separation_prior
title: "Heterochrony-Aware Prediction-Derived Separation Prior"
status: refined
added: 2026-07-13T13:54:28+08:00
---

# Heterochrony-Aware Prediction-Derived Separation Prior

## Thesis

Model pediatric epiphyseal separation as a conditional relation distribution rather than a fixed shape atlas. Global radiographic development and candidate-triggered local ossification evidence condition a frozen relation posterior, which regularizes arbitrary segmentation logits through a reversed-topology max-min bridge energy.

## Key design

- no fixed component count, named slot, dense prior map, or foreground-connectivity objective;
- prediction-derived candidates and local evidence that cannot create or move nodes;
- class-conditional `p(u | y_sep,c)` with aleatoric variance;
- context-only epistemic gate that cannot veto abnormal prediction geometry;
- synthetic bridge/over-split plus patient-disjoint OOF prediction training;
- audit-only violation maps and shared logits-space integration.

## Grounding papers

- [[carneiroesteves2024_plugandplay_framework_curvilinear]]
- [[wyburd2024_anatomically_plausible_segmentations]]
- [[zhao2024_adaptive_fusion_deep]]
- [[wu2024_deep_closing_enhancing]]
- [[you2024_learning_explicit_shape]]
- [[li2025_topology_optimization_medical]]
- [[lu2026_pgrnet_priorguided_roi]]

## Status

Method refinement READY at 9.25/10. No experiments were run. Full proposal: `research-workflow/refine-logs/FINAL_PROPOSAL_20260713_135428.md`.
