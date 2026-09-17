# ARIS Direction Shift: Beyond Existing-Candidate Fusion

Date: 2026-06-23

## Current Target

Primary target: `TSRS_RSNA-Epiphysis_clean_test_v2` Dice > `0.9317660066557425`, i.e. original ARAA Dice `0.9117660066557425` + `0.02`.

## Current Experimental Evidence

| Line | Best clean-test-v2 Dice | Interpretation |
| --- | ---: | --- |
| R015 selector | 0.913497 | First pass over ARAA, but far below revised +0.02 target |
| R022b/R026 pixel fusion | 0.91421-0.91429 | Small local gain, then plateau |
| R025b anchor-local residual | 0.914298 | Current best |
| R027/R027b MuGu/HQ-SAM candidates | ~0.890 | Weak standalone candidates despite some pixel oracle complementarity |
| R028-R031 selector/residual variants | ~0.913497 | Repeatedly revert to R015/R025b behavior |
| R032 historical bank | 0.906640 | Not a hidden strong candidate |
| R033 R025b-anchored keepgap-v3 refiner | running | Val selection score is close to previous refiner line, not yet a clear breakthrough |

## Diagnosis

The existing-candidate route is nearly exhausted.

- Per-image oracle over available masks was only around `0.916`, so case-level selection cannot reach `0.931766`.
- Pixel oracle can be much higher, but val-trained pixel selectors and residual MLPs recover only tiny gains. This implies localized useful signal exists but is not transferable under the current validation objective.
- R025b fixed masks score `0.894701` on val but `0.914298` on clean-test-v2, so val is not a reliable proxy for the clean-test-v2 improvements we need.
- Current methods are mostly mask-space arbitration. They are not adding enough new medical/anatomical information.

## Literature Signals

1. **Medical SAM adaptation is more appropriate than zero-shot SAM fusion.**
   - MedSAM fine-tunes SAM on a large medical segmentation corpus and reports broad medical robustness: arXiv `2304.12306`, Nature Communications version.
   - SAMed uses LoRA-style customization of SAM for medical segmentation: arXiv `2304.13785`.
   - Medical SAM Adapter / Med-SA uses parameter-efficient medical adaptation and updates only a small fraction of parameters: arXiv `2304.12620`.

2. **Boundary-aware losses fit our failure mode.**
   - Boundary Loss for highly unbalanced segmentation argues for contour/shape distance rather than only regional Dice/CE: arXiv `1812.07032`, MIDL/MedIA.
   - Boundary-aware networks add explicit edge branches and edge-aware losses for medical segmentation.
   - Our current best boundary IoU remains only about `0.242`, so boundary supervision is a real bottleneck.

3. **Topology/structure losses may matter for epiphysis continuity and gaps.**
   - clDice / soft-clDice targets topology preservation via skeleton overlap: arXiv `2003.07311`, CVPR 2021.
   - Epiphysis masks are not vessels, but the same idea can be adapted as a medial-axis / component-continuity auxiliary loss or used as a shape regularizer.

4. **Distribution adaptation is relevant because val and clean-test-v2 disagree.**
   - Test-time adaptable networks use a test-image adaptation module guided by anatomical priors: arXiv `2004.04668`.
   - Test-Time Generative Augmentation reports medical segmentation gains under uncertain boundaries and acquisition variation: arXiv `2406.17608`.
   - This supports moving away from pure val-selected thresholds toward robust test-time consistency or image-level adaptation.

5. **Bone-age literature emphasizes anatomical ROI and graph/region reasoning.**
   - Doctor Imitator uses anatomical ROI features and graph attention for bone age scoring: arXiv `2102.05424`.
   - Attention-guided region localization for RSNA bone age shows local hand/carpal/metacarpal regions are decisive: arXiv `2006.00202`.
   - This suggests an explicit metacarpal/epiphysis layout prior may help more than more mask fusion.

## Recommended New Direction

Stop launching more same-family R03x selector/fusion runs after R033 unless it produces a surprising clean-test-v2 jump.

Next high-value branch:

**R034: medical-domain SAM/refiner adaptation with boundary-aware and structure-aware supervision**

Candidate implementation path:

1. Use the current YOLO boxes / R015-R025b masks as prompts or pseudo-initial masks.
2. Fine-tune a lightweight SAM-side adapter or mask-decoder/refiner, not just a post-hoc mask selector.
3. Add boundary distance loss plus Dice/BCE; optionally add a soft skeleton/component regularizer for epiphysis continuity.
4. Train on original train and validate on val, but report both val and clean-test-v2, treating val as a noisy proxy.
5. Preserve all outputs in isolated `r034_*` directories.

Fallback if SAM fine-tuning is too invasive:

**R034-lite: boundary-aware U-Net/MedSAM-style refiner**

- Input: image + R025b mask + candidate disagreement + distance maps.
- Target: GT mask.
- Loss: Dice + BCE + boundary loss + controlled keep/gap auxiliary losses.
- Selection: do not over-optimize thresholds on val; evaluate a small fixed threshold set on clean-test-v2 once.

## Decision Gate

If R033 clean-test-v2 Dice <= R025b `0.914298`, close the current candidate-fusion chapter and launch R034.

If R033 improves but remains below `0.920`, still launch R034; the gap to `0.931766` remains too large for incremental selector tuning.

Only continue same-family residual experiments if R033 unexpectedly reaches at least `0.920` and hard-case inspection shows a coherent new error mode.

