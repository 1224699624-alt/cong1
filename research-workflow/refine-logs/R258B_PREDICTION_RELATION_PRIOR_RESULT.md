# R258B Prediction-Derived Relation Prior Result

## Decision

`no_go_relation_prior_not_validated`

The fixed 2-variant x 3-seed x 4-epoch experiment completed on all prediction-derived train/original-val pairs. `clean-test-v2` and Articular-Surface were not used. All six checkpoint hashes and the pair-manifest hash were independently reproduced.

## Three-seed fixed-epoch means

| Metric | Image + centers | + age/sex | Delta |
|---|---:|---:|---:|
| AUROC | 0.950054 | 0.953284 | +0.003230 |
| Average precision | 0.933825 | 0.939086 | +0.005261 |
| Accuracy | 0.882452 | 0.886952 | +0.004499 |
| Positive recall | 0.870266 | 0.907762 | +0.037495 |
| Close-pair recall | 0.954048 | 0.972283 | +0.018235 |
| Contact-pair recall | 0.962240 | 0.979167 | +0.016927 |
| Negative specificity | 0.894638 | 0.866142 | -0.028496 |
| Separation heat Dice | 0.319596 | 0.339317 | +0.019720 |

Heat-target coverage was `0.991564`. The training set contained 14,491 positive and 14,491 matched negative pairs over 871 usable cases; original-val contained 1,778 positive and 1,778 negative pairs over all 96 cases.

## Gate interpretation

The development branch failed two gates:

- minimum added value: AUROC delta `+0.003230 < +0.005` and close-recall delta `+0.018235 < +0.02`;
- specificity noninferiority: delta `-0.028496 < -0.02`.

Case-bootstrap confirms the trade-off rather than random noise:

- close-recall mean delta `+0.011801`, 95% CI `[+0.007659, +0.016830]`;
- specificity mean delta `-0.025172`, 95% CI `[-0.035518, -0.015486]`;
- AUROC delta is inconclusive, 95% CI `[-0.012732, +0.016572]`.

Real metadata still beats shuffled metadata in AUROC (`0.953284` vs `0.946406`), while metadata-only is chance (`0.499993`). X-ray and center-map ablations collapse AUROC to `0.561675` and `0.599277`. Thus age/sex contains a genuine conditional signal, but the current additive conditioning uses it mainly as extra separation pressure and produces too many false separation decisions.

## Next justified experiment

Do not integrate this branch into nnU-Net or use clean-test-v2. A follow-up should preserve the positive close/contact signal while explicitly controlling false separation, for example a residual/gated metadata modulation initialized at zero plus a stronger negative-specificity or asymmetric false-separation penalty. It must use a new preregistered fixed protocol rather than post-hoc threshold search on this result.
