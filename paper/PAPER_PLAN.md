# AAAI 2027 Draft Plan

## One-Sentence Contribution

Instance-derived background seam supervision reduces component under-counting between developing pediatric hand bones while preserving nearly the same semantic Dice, although unweighted seam classification introduces surface-distance outliers.

## Claims and Evidence

| Claim | Evidence | Status |
|---|---|---|
| Indexed instance labels can generate a training-only seam target without inference-time GT | R271 target construction and R274/R275 inference signature | Supported |
| Seam supervision changes the adjacent-structure error profile | CMR 0.5104 to 0.4271; gap FP 0.1873 to 0.1832 | Supported on nnU-Net/original-val |
| Overlap remains essentially unchanged | Dice 0.8980 to 0.8974 | Supported |
| The current loss uniformly improves boundary geometry | HD95 and ASSD worsen | Not supported |
| The method is plug-and-play across backbones | Only nnU-Net has mature matched results | Not yet supported |
| Age/sex conditioning improves separation | No matched completed experiment | Not yet supported |

## Main Figures and Tables

1. Figure 1: full-image and close-gap comparison for case 10375.
2. Table 1: matched R201 overlap, boundary, surface, gap, and component metrics.
3. Future submission table: U-Net, multiscale decoder, and nnU-Net with control / separation loss / prior+loss.

## Submission-Readiness Gates

- Add repeated-seed uncertainty or paired bootstrap confidence intervals.
- Complete a decoupled weighted separation loss with foreground support.
- Evaluate the identical plug-in objective on at least three backbones.
- Add age/sex and gap-width subgroup analysis before claiming demographic conditioning.
- Freeze the method before one final clean-test-v2 evaluation.
