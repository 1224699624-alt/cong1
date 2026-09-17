---
type: experiment
node_id: exp:filtered_trainval_variant_control
added: 2026-06-15T13:25:00Z
---

# Experiment: filtered train-val variant control

## Summary

This page tracks the project rule that dataset cleaning experiments must stay isolated from the default dataset and default experiment line.

## Isolated launchers already present locally

- `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_badlabel_filtered_v1.sh`
- `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_trainval_filtered_v1.sh`

## Purpose

- test whether removing suspicious train/val labels improves the same refiner family
- prevent data-cleaning gains from being misreported as pure architecture gains

## Project interpretation

- filtered-data runs are valid and useful
- but they must remain a separate evidence track from the unfiltered default benchmark line
- any win here should be reported as "method + filtered data" rather than silently replacing the standard baseline

## Connections

[AUTO-GENERATED from graph/edges.jsonl - do not edit manually]
