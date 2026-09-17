# R181 Candidate Pool Decision Gate

Date: 2026-07-03

## Purpose

After R179 improved topology but failed Dice, R181 checked whether the older SAM/HQ-SAM/MedSAM candidate pool still contains enough non-leaking complementarity to justify another fusion, readout, or arbitrator training run.

This was a read-only server audit. It did not train, tune on clean-test-v2, or use the new/reannotated test split.

## Compared Sources

All systems were evaluated on `TSRS_RSNA-Epiphysis_clean_test_v2/test` using existing prediction masks:

- `r110`: current valid best
- `r027_hq_gated`: MuGu-style SAM -> HQ-SAM gated source
- `r027b_hq_always`: always-HQ-SAM source
- `r036_medsam`: MedSAM+GBC source
- `r095b_r036_patch`: previous R036 patch arbitrator
- `r115_sam_fgbg`: SAM foreground/background precision adapter

Artifact:

- `outputs/analysis/r181_sam_hq_medsam_candidate_pool_audit.json`

## Key Results

Mean clean-test-v2 Dice:

- R110: `0.917723`
- R095b/R036 patch: `0.916641`
- R027b always HQ-SAM: `0.890792`
- R027 gated HQ-SAM: `0.890135`
- R036 MedSAM+GBC: `0.817558`
- R115 SAM fgbg adapter: `0.632215`

Per-image oracle over the whole pool:

- Dice: `0.918900`
- gain over R110: `+0.001177`
- best-source selections: R110 `53/81`, R095b `25/81`, R027 gated HQ-SAM `3/81`

Naive combinations:

- union Dice: `0.804319`
- intersection Dice: `0.665740`

## Interpretation

The pool has small per-image complementarity, mostly from the already-tested R095b branch. The gain ceiling is far below the target `0.931766`, and naive union/intersection are destructive. This is not enough to justify another GPU readout/fusion/arbitrator run.

R181 closes these routes:

- no R182 fusion over R110/R027/R027b/R036/R095b/R115;
- no renewed SAM/HQ-SAM/MedSAM candidate exploitation without a genuinely new source;
- no additional DINOv3 instance-separation loss-weight sweep after R179/R180.

## Next Valid Routes

1. R168/R170 reviewed train/val label-protocol route once `r168_reannotation_protocol_review_reviewed.csv` exists.
2. A genuinely new model family or environment upgrade, not based on the exhausted candidate-fusion pool.
3. A paper/patent claim can use R176/R179 as evidence that topology metrics are measurable and improveable, but not as evidence that the current model beats SOTA on all metrics.

