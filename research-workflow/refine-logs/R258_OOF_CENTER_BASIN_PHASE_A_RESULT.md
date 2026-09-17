# R258 OOF Center/Basin Proposal Phase A Result

## Decision

`allow_r258_phase_b_relation_prior`

R258 Phase A completed on all 875 `TSRS_RSNA-Epiphysis_contrast_v1` training images. Every image was inferred exactly once by a fold model that did not train on that image. `clean-test-v2` and `TSRS_RSNA-Articular-Surface` were not used.

## Aggregate OOF result

| Gate metric | Result | Required | Pass |
|---|---:|---:|:---:|
| Images processed exactly once | 875/875 | 875/875 | yes |
| Center recall | 0.976284 | >= 0.85 | yes |
| Center precision | 0.935650 | >= 0.75 | yes |
| Close-pair coverage | 0.953295 | >= 0.80 | yes |
| Contact-pair coverage | 0.920724 | >= 0.75 | yes |
| Global basin-scale error | 0.223896 | <= 0.50 | yes |
| Close endpoint basin-scale error | 0.188465 | <= 0.60 | yes |
| Contact endpoint basin-scale error | 0.170505 | <= 0.60 | yes |
| Proposal count MAE | 1.603429 | <= 3.0 | yes |
| False proposals/image | 1.744000 | <= 3.0 | yes |

The OOF detector matched 22,188 of 22,727 GT centers from 23,714 proposals. It covered 7,899 of 8,286 close pairs and 2,288 of 2,485 contact pairs.

## Training stability and integrity

- All five fixed eight-epoch folds completed with finite, decreasing training loss.
- The repaired fold 4 decreased from `8.068834` to `0.991221`; no Infinity, NaN, traceback, or OOM occurred.
- Three result CSV SHA256 values and all five fold-checkpoint SHA256 values were independently reproduced.
- Fold and inference manifests each contain 875 unique image identities exactly once.
- No checkpoint from the rejected non-finite run was reused.

## Caveat for Phase B

Fold 4 has substantially larger basin-scale error than folds 0--3: global `0.404373`, close `0.566047`, and contact `0.552529`. These remain inside the preregistered thresholds but indicate fold-dependent scale noise. Phase B should use the fixed OOF proposals without fold-specific tuning, report fold-stratified results, and avoid treating predicted scale as an exact geometric measurement.

## Next permitted step

Implement and train the matched Phase B relation-prior comparison:

1. local X-ray plus two prediction-derived center maps;
2. the same inputs plus continuous bone age and sex;
3. no explicit geometry vector;
4. fixed training/evaluation protocol with original-val only for the downstream audit.

This Phase A result does not yet establish a segmentation improvement and does not permit clean-test-v2 evaluation or nnU-Net integration.
