# Round 2 Refinement: Prior-Calibrated Reliable Negative Reweighting Loss (PCR-Loss)

## Problem Anchor

- Bottom-line problem: 在不依赖高质量骨缝标签、不构造反向背景 seam GT 的前提下，利用发育先验调整训练置信和损失系数，使不同分割模型更容易区分相邻骨头。
- Must-solve bottleneck: 当前监督是骨头 mask，边界和局部组件存在标注噪声；直接反转 mask 会把骨头标注误差完整复制成错误骨缝监督，而普通 Dice/BCE 又会平等拟合错误边界和真实骨头内部。
- Non-goals: 不生成伪骨缝 mask；不自由删除前景；不把模型自身高置信当作标签必然正确；不增加第二分割器；不使用 clean-test-v2 调权重。
- Constraints: 仅 `TSRS_RSNA-Epiphysis`；R201；原始数据不变；同一先验与 loss 规则跨模型；original train/val 开发；clean-test-v2 最终一次评估。
- Success condition: 在 Dice/IoU/Recall 非劣的同时，跨多个模型降低 merge rate、gap-region FP 和 component-count MAE，并改善 Boundary IoU/F1、Surface Dice、HD95/ASSD。

## Final Method

### 1. Reliability-weighted noisy-label base loss

```text
C_fg = erosion(y,r_core)
C_bg = outside(dilation(y,r_far))
U_bd = remaining near-boundary pixels
```

```text
q_label = q_min + (1-q_min)*q_ann_morph
q_x = stopgrad(q_label(x)*(1-delta*u_model(x)))
0 <= delta <= fixed delta_max
```

`q_ann_morph` 只来自 morphology 与真实存在的独立标注一致性；OOF/augmentation uncertainty 只能有界弱衰减，不能判断标签正确，也不能通过 teacher-label agreement 提高权重。

```text
L_pix = separately_normalized_mean_{C_fg,C_bg}
        q_x*BCE(p,y)
      + normalized_mean_{U_bd}
        q_x*GCE_gamma(p,y)

L_wDice = 1 - (2 sum q_x p y + eps)
                  /(sum q_x p + sum q_x y + eps)

L_base = L_wDice + L_pix
```

BCE 与 GCE 的区域互斥；不再增加 teacher consistency。

### 2. Prior only produces a pair coefficient

```text
k_ij = stopgrad(
       P(y_sep=1 | image context,prediction geometry)
       * c_epi(context only)
       * c_candidate
       * c_stability)
```

prior 不产生 seam、corridor target 或 cut，只决定已有负监督的重要程度。

### 3. Valid pair and legal negative anchors

训练期 candidate 必须先由模型 prediction 产生；随后才能用 GT 匹配到两个不重叠 eroded foreground supports。两 support 的 between-region 内：

```text
N_ij = {x: y(x)=0,
           x in between-region(i,j),
           q_label(x)>=q_floor}
```

若原 binary GT 没有任何 `y=0`，该 pair 严格跳过。GT match 和 `N_ij` 仅存在于训练 loss，推理 candidate generation 不使用 GT。

### 4. Stop-gradient hard-negative coefficient

在每个 `N_ij` 内，按当前 foreground probability 选择 top-k hard negatives；索引 stop-gradient。定义：

```text
h_ij(x) in [0,1]    # stop-gradient top-k indicator/soft rank weight
```

多个 pair 覆盖同一像素时使用有界 max，不做 sum：

```text
a_x = max_{ij:x in N_ij} [k_ij*h_ij(x)]
0 <= a_x <= 1
```

无 pair 覆盖时 `a_x=0`。最终 pair loss 只按实际有效权重质量归一化：

```text
Z_pair = sum_{x:y_x=0} q_label(x)*a_x

if Z_pair == 0:
    L_pair = 0
else:
    L_pair = - sum_{x:y_x=0}
             q_label(x)*a_x*log(1-p_x)
             /(eps + Z_pair)
```

因此：

- target 方向仍完全来自原 bone mask 的 `y=0`；
- prior 只调系数；
- hard prediction 只决定优化顺序，不决定标签；
- 重叠 pair 不会重复放大像素；
- pair loss 不受图像尺寸、远背景面积、有效 pair 数和 top-k 数任意稀释。

### 5. Final two-term cross-model loss

```text
g_b = ||dL_base/dl||_2
g_p = ||dL_pair/dl||_2

if Z_pair==0 or g_p<tau_g:
    L = L_base
else:
    alpha = clip(
        rho*stopgrad(g_b/(g_p+eps)),
        alpha_min,alpha_max)
    L = L_base + alpha*L_pair
```

- 只在最终 mask logits 上计算 norm；detection/objectness loss 不进入 ratio。
- `alpha` stop-gradient，不涉及二阶梯度。
- 同一 reliability construction、top-k、pair aggregation、gradient ratio 和 warm-up 在 original train/val 冻结并跨 backbone 共用。

## Capability Boundary

- PCR-Loss 可以加强原标签已经存在但被模型误预测为前景的骨间负像素，并降低噪声边界的错误梯度。
- 它不能恢复训练标签中完全不存在的骨缝。如果 GT 将两骨完整连成一个前景且没有 `y=0`，必须 skip。
- 这不是 reverse-background labeling；它是原 bone-mask supervision 下的可靠性与 hard-negative gradient allocation。

## Minimal Validation Boundary（当前不执行）

- Native Dice/BCE vs piecewise robust base vs reliability base vs full PCR-Loss。
- Uniform hard-negative weight vs prior-calibrated coefficient。
- 报告 valid-pair coverage、selected-negative count、merge/gap/component metrics、Boundary metrics、Recall/Dice 非劣和 harmful activation。
- 同一冻结 prior/loss 规则接入至少 5 个模型；clean-test-v2 仅冻结后 R201 final evaluation。

