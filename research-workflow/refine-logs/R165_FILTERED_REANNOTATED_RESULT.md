# R165 Filtered Reannotated Result

## Status

DONE_NEGATIVE.

R165 trained the DINOv3 instance-separation source model on the R164 filtered reannotated train/val variant:

`TSRS_RSNA-Epiphysis_reannotated_filtered_trainval_v1`

Success evaluation used only:

`TSRS_RSNA-Epiphysis_clean_test_v2/test`

The reannotated test split was not used.

## Training Result

- Epochs: `28`
- Best validation epoch: `20`
- Best validation Dice: `0.8364260441848943`
- Best validation Precision: `0.8124008482919286`
- Best validation Recall: `0.8811247268574027`
- Best validation Boundary IoU: `0.4407166633400138`
- Best validation false-bridge flag: `0.18947368421052632`
- Threshold: `0.55`
- Separation suppression weight: `0.30`

## Final Evaluation

clean-test-v2/test:

- Evaluated images: `81`
- Dice: `0.8903028393595949`
- IoU: `0.8029255739205072`
- Precision: `0.878264802813878`
- Recall: `0.9050498031393276`
- Boundary IoU: `0.1849899312529587`
- Component-count error: `1.1851851851851851`
- False-bridge flag: `0.4444444444444444`

Original-test control:

- Evaluated images: `97`
- Dice: `0.8748391142656348`
- Precision: `0.8648729182360912`
- Recall: `0.8894627570207422`
- Boundary IoU: `0.1769552555504493`
- False-bridge flag: `0.422680412371134`

## Comparison

- R162 unfiltered reannotated clean-test-v2 Dice: `0.8726263750014701`
- R165 filtered reannotated clean-test-v2 Dice: `0.8903028393595949`
- Delta vs R162: `+0.017676464358124755`
- Current valid best R110 Dice: `0.9177231563529792`
- Delta vs R110: `-0.02742031699338432`
- Target Dice: `0.9317660066557425`
- Gap to target: `0.04146316729614763`

## Interpretation

Filtering the most obvious reannotated train outliers helped substantially relative to R162, which supports the R163 diagnosis that raw reannotated train labels contained damaging protocol/outlier noise. However, the filtered-data source model still remains far below R110 and the target.

The data-cleaning branch improves a weak reannotated run but does not create a competitive source model. The result is also below R143/R130/R117-style source runs in the broader history.

## Decision

Do not continue with small threshold tweaks to R164/R165 filtering. The observed gain is real but not close enough to the target, and the best validation Dice remains too low.

Next useful ARIS options:

1. Close the reannotated direct-training branch and return to a stronger architecture/environment route.
2. If data work continues, require manual correction or a more principled label-protocol reconciliation, not automated threshold filtering.
3. If GPU work continues immediately, it should be a materially new mechanism, not R165 with slightly different filter thresholds.
