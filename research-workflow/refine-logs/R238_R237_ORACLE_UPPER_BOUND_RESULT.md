# R238 R237 Oracle Upper-Bound Result

## ARIS Status
- Stage: original-val oracle upper-bound and selector audit
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: yes, isolated oracle-upper-bound output only
- Decision: `candidate_space_strong; selector_evidence_weak; build_non_oracle_generator_next`

## Purpose
R237 showed that background-connectivity seam cuts can safely improve boundary,
gap, and component-count diagnostics when the candidates are generated with GT
instance knowledge. R238 asks two follow-up questions:

1. Can R237 candidate rows be ranked with GT-free features?
2. If we replay the best R237 candidate per image, what is the mask-level upper
   bound this route can realistically target?

This is still original-val development. It is not clean-test-v2 evidence.

## Outputs
- Candidate gate script: `scripts/train_r238_r237_candidate_gate.py`
- Oracle replay script: `scripts/run_r238_r237_oracle_best_mask_editor.py`
- Visual audit script: `scripts/build_r238_oracle_best_visualizations.py`
- Gate audit JSON: `outputs/analysis/r238_r237_candidate_gate_groupedcv.json`
- Gate audit rows: `outputs/analysis/r238_r237_candidate_gate_groupedcv_rows.csv`
- Oracle replay summary: `outputs/analysis/r238_r237_oracle_best_mask_editor_summary.json`
- Oracle replay per-image CSV: `outputs/analysis/r238_r237_oracle_best_mask_editor_per_image.csv`
- Oracle replay selected CSV: `outputs/analysis/r238_r237_oracle_best_mask_editor_selected.csv`
- Oracle masks: `outputs/ablations_variants/r238_r237_oracle_best_mask_editor/TSRS_RSNA-Epiphysis/val/masks`
- Visual audit: `outputs/analysis/r238_r237_oracle_best_visual_audit/index.html`

## R238-F0 Candidate Gate Audit
Grouped-CV over R237 bounded candidate rows:

- rows: `747`
- images: `62`
- GT-free numeric features: `14`
- positive labels (`safe_nonmerge_gain`): `735`
- over-erosion proxy labels: `0`
- positive rate: `0.983936`
- risk rate: `0.000000`
- CV AUC: `0.110884`

Interpretation: the R237 bounded candidate set is already extremely clean under
GT diagnostics, so the grouped-CV gate is not very informative. It does not
prove that a deployable selector exists. The useful conclusion is weaker but
important: the candidate family is promising, while the current selector audit
does not contain enough hard negatives to validate robust GT-free selection.

## R238-A Oracle-Best Mask Replay
R238-A replays the best R237 safe candidate per image and writes isolated masks.
This is an oracle upper bound because candidate labels come from GT-derived
diagnostics.

Full original-val result over `96` R110 anchors:

| Metric | Mean delta |
| --- | ---: |
| Edited images | `61/96` |
| Dice | `+0.000120` |
| IoU | `+0.000201` |
| Precision | `+0.000256` |
| Recall | `+0.000000` |
| Boundary IoU | `+0.000882` |
| Boundary F1 | `+0.001164` |
| gap-region FP rate | `-0.000657` |
| component count MAE | `-0.041667` |
| mean cut pixels | `17.427083` |

Compared with R236:

| Run | Edited | Dice Δ | IoU Δ | Recall Δ | Boundary IoU Δ | Boundary F1 Δ | Gap FP Δ | Component MAE Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R236 loose-prob HGB | `20/96` | `+0.000046` | `+0.000073` | `+0.000000` | `+0.000149` | `+0.000198` | `-0.000176` | `+0.000000` |
| R238-A oracle upper bound | `61/96` | `+0.000120` | `+0.000201` | `+0.000000` | `+0.000882` | `+0.001164` | `-0.000657` | `-0.041667` |

R238-A is clearly stronger than R236 as an upper bound, especially on edited
coverage, boundary metrics, gap FP, and component count MAE.

## Visual Audit
Generated `61` original-val panels:

- `outputs/analysis/r238_r237_oracle_best_visual_audit/index.html`

Top metric-positive cases include:

- `3468.png`: Dice `+0.000819`, Recall `+0.000000`, Boundary IoU `+0.004854`, gap FP `-0.003198`
- `6606.png`: Dice `+0.000455`, Recall `+0.000000`, Boundary IoU `+0.004823`, gap FP `-0.002995`
- `3840.png`: Dice `+0.000474`, Recall `+0.000000`, Boundary IoU `+0.004438`, gap FP `-0.002513`
- `1620.png`: Dice `+0.000393`, Recall `+0.000000`, Boundary IoU `+0.003522`, gap FP `-0.002337`
- `3602.png`: Dice `+0.000581`, Recall `+0.000000`, Boundary IoU `+0.003480`, gap FP `-0.002826`

The generated panels should be inspected before any model claim. The numerical
diagnostics say these edits do not remove GT foreground under the oracle replay,
but visual review is still required because the final project target explicitly
requires avoiding over-erosion and漏分.

## Interpretation
R238 strengthens the background-connectivity route:

- The upper bound beats R236 on the exact diagnostics we care about.
- Recall remains unchanged, and Dice/IoU move slightly positive.
- Component count MAE finally moves in the desired direction.
- The evidence is still not deployable because the candidate source/labels use
  GT instance information.

The main blocker is now not whether seam-background edits can help; it is how to
generate/select the same style of candidates without GT.

## Next Step
Build `R239` as a non-oracle candidate generator that targets the R238-A edit
profile:

- reuse GT-free seam probability from R236/R228;
- add background-channel geometry features from R237;
- generate candidates from predicted connected components and local background
  seeds, not GT instance pairs;
- train or gate against R237/R238 labels on original train/val only;
- require original-val mask-level validation against R236:
  - edited coverage substantially above `20/96`;
  - Recall delta `0`;
  - Boundary IoU/F1 and gap FP better than R236;
  - component count MAE non-worsening, ideally negative;
  - visual audit showing seam cleanup without over-erosion.

Do not run clean-test-v2 until a non-oracle R239/R240 variant passes these gates.
