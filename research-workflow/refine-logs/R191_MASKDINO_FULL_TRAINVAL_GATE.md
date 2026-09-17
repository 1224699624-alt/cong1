# R191 MaskDINO Full Train/Val Gate

Server-only run on `/home/shenzeyu/workspace/YOLO_SAM_generic_src` using isolated env `aris-maskdino-r189-cu130`.

## Setup

- data: original `TSRS_RSNA-Epiphysis` train/val only
- converted to COCO instance format at max size 384
- train images: full original train
- val images: full original val
- clean-test-v2: not used
- new/reannotated test: not used
- checkpoint init: empty `MODEL.WEIGHTS`
- iterations: 300

## Result

- status: `train_and_eval_completed` is not applicable here; see `outputs/analysis/r191_maskdino_full_trainval_gate/r191_summary.json` for raw R191.
- R191 training completed and saved `model_final.pth`.
- COCO validation AP remained zero for bbox and segm.

## Decision

R191 proves the faithful MaskDINO stack can train on the server, but 300 iterations from empty weights did not learn usable spatial predictions. R192 diagnostic is required before any clean-test-v2 evaluation.
