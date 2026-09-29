# DualPriorSeg paper fill-in checklist

Remove or resolve every `[TBD ...]` marker before submission.

## Data and citations
- Insert the formal TSRS dataset citation.
- Freeze and report the final TSRS train/val/test counts.
- Freeze and report the exact RAM-W600 split used by the released experiment manifest.
- Verify all reference years/venues against the final bibliography format.

## Figures
- Figure 1: representative TSRS seam failure + RAM overlap failure, with matched backbone and DualPriorSeg predictions.
- Figure 2: final architecture diagram matching the released code path.
- Figure 3: qualitative success and failure cases with error overlays; no test-set cherry-picking.

## Final experiment block
- Hardware, software, batch size, learning rates, backbone LR ratio, epoch budget, augmentation, checkpoint-selection rule, random seeds.
- State clearly whether the final paper uses adapter-only, joint fine-tuning, or both.

## Tables
- Table 1: final TSRS + RAM main results.
- Table 2: nnU-Net native-continue / seam-only / overlap-only / dual ablation.
- Table 3: nnU-Net / TransUNet / Swin-UMamba base vs +DualPriorSeg portability.
- Table 4: official RAM-W600 benchmark comparison using the exact published metric definitions.

## Statistical reporting
- Prefer at least 3 seeds for the selected final configuration.
- Report mean ± standard deviation or paired bootstrap confidence intervals where appropriate.
- Bind each final result row to one checkpoint epoch/hash; do not mix best metrics from different epochs.

## Claims
- Use `competitive with state-of-the-art architectures` unless official matched test results clearly justify a stronger claim.
- Do not claim that gradient routing guarantees non-degradation.
- Do not claim TSRS true overlap supervision unless multi-membership ground truth becomes available.
- Do not claim backbone-agnostic generalization until cross-backbone experiments are completed.
