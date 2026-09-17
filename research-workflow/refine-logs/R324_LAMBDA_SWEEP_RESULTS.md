# R324 RAM-W600 overlap-prior lambda sweep

## Status

- Completed normally on 2026-08-03.
- Server: westd port 27945, RTX 4080 SUPER.
- No active screen session; GPU is idle after completion.
- The test split was not used.
- All arms share the mature R285 nnU-Net initialization, RAM-W600 train/val split, seed 3241, threshold 0.5, relation graph, checkpoint rule, and evaluation protocol.
- Prior arms stopped at epoch 16 under `min_epochs=15` and `patience=10`; the selected checkpoint was epoch 6.

## Results

| Lambda | Macro DSC | Macro IoU | Overlap DSC | Overlap IoU | Overlap NSD@2px | Overlap MSD px (lower is better) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.978083372 | 0.957320392 | 0.872430693 | 0.773727131 | 0.879367818 | 1.129427633 |
| 0.001 | 0.978086591 | 0.957326293 | 0.872449706 | 0.773757039 | 0.879273827 | **1.129110144** |
| **0.002** | **0.978095233** | **0.957342386** | **0.872682626** | **0.774123522** | **0.879411574** | 1.129845896 |
| 0.004 | 0.978083253 | 0.957319558 | 0.872599117 | 0.773992108 | 0.878985727 | 1.131017007 |
| 0.006 | 0.978061974 | 0.957279027 | 0.872491175 | 0.773822274 | 0.878618030 | 1.133695117 |
| 0.010 | 0.977947712 | 0.957062602 | 0.871889566 | 0.772876315 | 0.877261433 | 1.143955444 |

## Decision

- `lambda=0.002` is the best balanced configuration: all overlap area metrics and NSD improve together, while Macro DSC and Macro IoU also remain above the plain baseline.
- `lambda=0.001` minimizes overlap MSD but does not maximize the other metrics.
- `lambda=0.004` is the turning point. It still helps Overlap DSC/IoU slightly, but global DSC/IoU and boundary metrics cease improving.
- `lambda>=0.006` over-constrains the model. Stronger intervention produces larger local changes but worsens average boundary quality and eventually the global metrics.
- This is a single-seed validation result and supports a candidate-setting decision, not a final statistical claim.

## Why the run is fast

This run fine-tunes a mature R285 nnU-Net instead of training the backbone or a separate prior network from scratch. The output-side IEM has no learnable inference branch: after the logits, it constructs sparse relation-violation masks and adds a local loss. Only about 0.069% of pixels are critical pixels. RAM-W600 contains 425 training images at 384 x 384, one epoch takes roughly 22--36 seconds on the RTX 4080 SUPER, the plain arm is reused, and early stopping ends each prior arm at epoch 16.

## Artifacts

- Numeric sweep: `outputs/visualizations/r324_lambda_sweep/lambda_metrics.csv`
- Per-case metrics: `outputs/visualizations/r324_lambda_sweep/case_overlap_metrics.csv`
- Aggregate plot: `outputs/visualizations/r324_lambda_sweep/lambda_metric_curves.png`
- Eight real-image comparison panels: `outputs/visualizations/r324_lambda_sweep/case_*.png`
