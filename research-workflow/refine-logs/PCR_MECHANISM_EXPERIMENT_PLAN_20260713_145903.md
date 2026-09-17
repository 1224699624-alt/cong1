# PCR-Loss Mechanism-First Experiment Plan

**Problem**: 验证 HA-PDSP 先验系数与 PCR-Loss 是否在多个普通分割骨架上，以可解释的训练动力学减少骨缝粘连和边界模糊。  
**Method thesis**: 先验不生成骨缝标签，只提高原 bone mask 中可靠骨间 hard negatives 的梯度权重；piecewise robust base loss 降低噪声边界的错误梯度。  
**Date**: 2026-07-13  
**Current action**: 只规划，不启动实验。

## Claim Map

| Claim | Minimum convincing evidence | Paper target |
|---|---|---|
| C1 Mechanism | PCR-Loss 训练时，骨间 selected negatives 的前景置信随 epoch 明显下降，而 bone core 置信和 Recall 不下降；pair gradient 与 base gradient 无持续严重冲突 | Training-dynamics figure + ablation table |
| C2 Transfer | 同一冻结 prior、reliability、top-k、gradient-ratio 规则在至少 4/5 个异构 backbone 上改善骨缝/边界，且不是单一 U-Net 特例 | Main cross-backbone table + qualitative grid |
| Anti-claim | 提升不是来自重新调阈值、额外训练预算、更多参数或选择最好看的 test case | Paired seeds、same recipe、fixed threshold report、pre-registered figure manifest |

## Backbone Strategy

### Development backbone

1. **Classic 2D U-Net**：结构简单、参数路径清晰，用于开发 loss 和解释训练动力学。优先复用 `scripts/train_context_segmenter.py`/现有 U-Net 数据与评估管线，但建立独立输出目录。

所有 PCR 超参数只允许在 U-Net 的 original train/val 上冻结一次。

### Locked transfer backbones

2. **nnU-Net 2D**：强医学分割基线；项目已有 R202 数据转换、launcher 和 R201 输出。R202 baseline 已达到 Dice `0.923573`、Boundary IoU `0.270431`、Boundary F1 `0.417773`，但 component merge rate `0.679012`，非常适合验证“总体分割强但仍会粘连”。
3. **DINOv3-UNet / timm encoder U-Net**：预训练表征型骨架，复用 `scripts/train_timm_unet_segmenter.py` 或现有 DINOv3 decoder。
4. **DeepLabV3+ 或可靠 ASPP-CNN**：非 U-shape CNN，用于证明不依赖 U-Net decoder；只有 faithful baseline 通过 overfit/val gate 后才进入主表。
5. **SegFormer-B0 或 YOLO+SAM trainable mask refiner**：Transformer/prompted-family 验证，放主表或 appendix，取决于 baseline 是否先达到可用质量。

不要为了凑模型数量使用已知训练崩溃的弱实现。主论文宁可使用 4 个可信模型，不使用 5 个失真的模型。

## Controlled Comparison Per Backbone

每个 backbone 至少运行同 seed 配对的四个版本：

| Variant | Purpose |
|---|---|
| B0 Native | 该模型原生 Dice/BCE 或 canonical nnU-Net loss |
| B1 Robust Base | `L_weightedDice + piecewise BCE/GCE`，无 pair prior |
| B2 Uniform Pair | B1 + `L_pair`，所有 valid pair 系数固定为 1 |
| B3 Full PCR | B1 + HA-PDSP prior-calibrated pair coefficient |

关键比较：B0→B1 证明 noisy-label allocation；B1→B2 证明 hard-negative mechanism；B2→B3 证明发育先验系数不是装饰。

## Training Fairness

- 同一 backbone 内：相同初始化 seed、数据顺序、增强、分辨率、batch、optimizer、LR schedule、epoch/early stop、checkpoint rule。
- 不同 backbone 之间：保留各自 canonical optimizer/patch/preprocessing，不强迫 nnU-Net 与 U-Net 使用相同 LR；唯一统一的是 PCR loss 规则和冻结超参数。
- 每个正式比较 `3 seeds`；先做单 seed gate，通过后再补其余种子。
- 主分析使用 canonical/fixed threshold；另报 original-val-selected threshold 作为 appendix，禁止 clean-test-v2 threshold search。
- 训练、模型选择和 sentinel analysis 仅 original train/val；clean-test-v2 只在全部冻结后一次性 R201 final evaluation。

## Training-Dynamics Instrumentation

### A. 每个 optimizer step 或每 N=50 steps 记录

| Group | Scalars |
|---|---|
| Optimizer | LR、weight decay、AMP scale、batch size、step time、GPU memory |
| Loss | `L_wDice`、core BCE、far-BG BCE、boundary GCE、`L_base`、`L_pair`、total loss |
| Gradient balance | `g_base`、`g_pair`、`alpha`、clip rate、`cos(g_base,g_pair)` on final logits |
| Reliability | mean/q10/q50/q90 of `q_label` and `q_x` in `C_fg/C_bg/U_bd`、fraction at `q_min` |
| Pair activity | candidate pairs、valid pairs、`Z_pair`、selected anchor count、`k_ij` quantiles、abstention rate |
| Confidence | mean p in bone core、uncertain boundary、far background、selected inter-bone negatives |

`cos(g_base,g_pair)<0` 表示 pair loss 与基础分割目标冲突；连续多轮强负值说明先验/anchor 选择有害。

### B. 每个 epoch 记录参数组统计

不保存每个权重的完整时间序列，按模块记录：

```text
||theta||2
||grad||2
||delta_theta||2 / (||theta||2 + eps)
gradient clipping fraction
activation mean/std/sparsity
```

分组至少包括：encoder early/mid/late、decoder blocks、final mask head。重点观察：

- PCR 是否只改变 final head，还是逐步改变 decoder/encoder；
- pair loss 启用后 update-to-weight ratio 是否突增；
- final head bias/logit distribution 是否出现全背景或过度切割倾向。

### C. 每个 epoch 的机制性验证量

```text
anchor_fp_confidence = mean p on selected y=0 hard negatives
core_confidence      = mean p on eroded foreground cores
confidence_gap       = core_confidence - anchor_fp_confidence
boundary_sharpness   = mean |grad p| near reliable GT boundary
pair_coverage        = valid pairs / proposed pairs
harmful_activation   = activated pairs followed by foreground-core recall loss
```

期望机制轨迹：`anchor_fp_confidence` 下降、`confidence_gap` 上升、core confidence/Recall 稳定、boundary sharpness 改善而非通过整体腐蚀获得。

## Sentinel Cohort and Epoch Visualizations

在训练前从 original val 建立固定 `24` 例 manifest，不按 PCR improvement 选择：

- 8 个窄骨缝/粘连高风险；
- 6 个边界模糊或低对比；
- 4 个薄小骨骺/Recall 风险；
- 4 个高 component error 风险；
- 2 个标签协议分歧案例。

按 image/GT-only 规则选择并冻结。每个 sentinel 在 epoch `0,1,5,10,best,last` 导出：

- X-ray；
- GT 与 reliability zones；
- baseline/PCR probability heatmap；
- binary mask；
- selected negative anchors 与 `k_ij`；
- FP/FN overlay；
- 边界/骨缝 zoom crop。

这些图用于观察训练演化，不用于阈值或超参数选择。

## Experiment Blocks

### Block 1: U-Net mechanism development — MUST-RUN

- Systems: B0/B1/B2/B3，先 1 seed，过 gate 后 3 seeds。
- Main question: reliability 与 prior coefficients 是否按预期改变梯度和局部置信。
- Go gate:
  - `L_pair` 非零覆盖足够（valid pair/anchor coverage 不能接近 0）；
  - selected-negative confidence 有持续下降；
  - Recall/core confidence 不明显下降；
  - 无 NaN、全背景、component explosion；
  - 至少在 original val 的 bridge/boundary sentinel 中看到一致方向，而非单例。
- Failure meaning: 若 coverage 太低，说明当前标签无法支撑 coefficient-only separation；不能靠调大 lambda 伪造效果。

### Block 2: U-Net ablation and causality — MUST-RUN

- B0→B1→B2→B3。
- Delete checks: 去 annotation reliability；去 bounded uncertainty；uniform vs prior coefficient。
- Main evidence: 训练动力学与结构指标共同表明 B3 不是简单增加背景权重。

### Block 3: Locked cross-backbone transfer — MUST-RUN

- nnU-Net、DINOv3-UNet、一个非 U-Net family；可用时再加 YOLO+SAM/SegFormer。
- 禁止 per-backbone PCR 超参搜索。
- Success: 至少 4/5 方向一致；如果只有 U-Net 改善，跨模型创新主张不成立。

### Block 4: Paper-ready qualitative evidence — MUST-RUN AFTER FREEZE

主图至少包含：

```text
Column 1: X-ray
Column 2: GT
Column 3: vanilla backbone prediction
Column 4: HA-PDSP + PCR-Loss prediction
Column 5: enlarged gap/boundary crop + error overlay
```

建议两张图：

1. **机制主图**：同一个 U-Net 病例，显示原图、GT、baseline、ours、可靠性图、selected anchors 和 zoom。
2. **跨模型定性图**：相同 2–3 个预注册病例，按行展示 U-Net/nnU-Net/DINOv3-UNet/第4模型的 baseline→ours。

图例固定：GT boundary 绿色、prediction boundary 洋红、FP 红色、FN 黄色。不得根据最终 improvement 最大值挑图；可展示预注册病例中的 median-improvement、典型成功和诚实失败各一例。

### Block 5: Failure and label-quality analysis — APPENDIX

- 完全无 `y=0` seam 的 GT：PCR 应 abstain，说明能力边界。
- label-disagreement/outlier subgroup。
- prior harmful activation、过分割和 Recall 下降案例。

## Quantitative Evidence Is a Guardrail, Not the Story

论文可以把视觉和训练机制放在中心，但不能完全忽略指标。最低限度应报告：

- Boundary IoU/F1、Surface Dice 2px/5px、HD95、ASSD；
- gap FP、merge rate、component-count MAE；
- Dice/IoU/Recall 非劣；
- per-image paired bootstrap 95% CI，3-seed mean±std；
- valid-pair coverage、activation/abstention/harmful activation。

指标用于证明图像不是 cherry-pick，主贡献仍可表述为结构性与可解释训练改善，而不是追求 Dice SOTA。

## Run Order

| Milestone | Goal | Runs | Decision Gate | Status |
|---|---|---|---|---|
| M0 | Instrumentation/data audit | reliability zones、valid pairs、sentinel manifest、logger dry-run | coverage/log schema/GT lock correct | TODO |
| M1 | U-Net paired baseline | U-Net B0 vs B3, 1 seed | training stable + mechanism trajectory positive | TODO |
| M2 | Causal ablation | U-Net B1/B2 + 3-seed completion | B3 adds value beyond robust base/uniform pair | TODO |
| M3 | Strong transfer | nnU-Net B0/B3, 1→3 seeds | no retune, same directional structure gain | TODO |
| M4 | Family transfer | DINOv3-UNet + non-U-Net B0/B3 | at least 4/5 positive | TODO |
| M5 | Protocol freeze | all thresholds/rules frozen | code/config hashes and manifest complete | TODO |
| M6 | Final R201 + figures | one locked clean-test-v2 evaluation | main table/figures/failure panel | TODO |

## Stop Rules

- U-Net valid-pair coverage/selected anchors 接近零：停止长训练，当前标签不支持该机制。
- `anchor_fp_confidence` 不下降但 core confidence 上升：检查 pair loss/anchor bug，不增加权重。
- Recall/core probability 明显下降或 components 爆炸：停止，说明变成了整体腐蚀。
- 只有开发 U-Net 提升，其他 backbone 无效：将论文主张降级为 U-Net-specific，不声称通用先验。
- 定性提升只出现在按结果选择的个别图：不构成证据。

## Paper Deliverables

- Table 1: 4–5 backbone baseline vs PCR-Loss，结构指标 + Dice/Recall guardrail。
- Table 2: U-Net B0/B1/B2/B3 ablation。
- Figure 1: PCR training dynamics（anchor probability、core probability、alpha、gradient cosine、pair coverage）。
- Figure 2: X-ray/GT/baseline/ours/zoom 的主定性图。
- Figure 3 or appendix: same cases across multiple backbones。
- Appendix: label-noise subgroup、abstention/failure cases、parameter-group dynamics。

