# Round 1 Refinement: Prior-Calibrated Reliable Negative Reweighting Loss (PCR-Loss)

## Problem Anchor

- Bottom-line problem: 在不依赖高质量骨缝标签、不构造反向背景 seam GT 的前提下，利用发育先验调整训练置信和损失系数，使不同分割模型更容易区分相邻骨头。
- Must-solve bottleneck: 当前监督是骨头 mask，边界和局部组件存在标注噪声；直接反转 mask 会把骨头标注误差完整复制成错误骨缝监督，而普通 Dice/BCE 又会平等拟合错误边界和真实骨头内部。
- Non-goals: 不生成伪骨缝 mask；不自由删除前景；不把模型自身高置信当作标签必然正确；不增加第二分割器；不使用 clean-test-v2 调权重。
- Constraints: 仅 `TSRS_RSNA-Epiphysis`；R201；原始数据不变；同一先验与 loss 规则跨模型；original train/val 开发；clean-test-v2 最终一次评估。
- Success condition: 在 Dice/IoU/Recall 非劣的同时，跨多个模型降低 merge rate、gap-region FP 和 component-count MAE，并改善 Boundary IoU/F1、Surface Dice、HD95/ASSD。

## Anchor Check

- prior 只产生 pair coefficient，不产生 seam/background target。
- hard target 始终是原 bone mask；训练 pair 必须包含原标签已有的背景像素，否则跳过。
- noisy-label robustness 与 pair separation 分工明确：前者调节全图监督可信度，后者只加强相邻候选间已有背景 hard negatives。

## Simplicity Check

- 删除 EMA consistency、valley ranking、全图 BCE 与 boundary GCE 重复、pair BCE/ranking 双实现。
- 最终只有 `L_base + alpha L_pair`。
- 不新增 trainable loss network；复用冻结 HA-PDSP relation posterior 输出 pair coefficient。

## Revised Method

### 1. Annotation-side label reliability

从原 bone mask 构造：

```text
C_fg = erosion(y,r_core)
C_bg = outside(dilation(y,r_far))
U_bd = remaining near-boundary pixels
```

annotation reliability 不与多项置信连乘：

```text
if independent second annotation exists:
    q_ann_morph = eta*q_annotation_agreement + (1-eta)*q_morph
else:
    q_ann_morph = q_morph

q_label = q_min + (1-q_min)*q_ann_morph
```

- `q_morph` 在 core/far background 高，在边界带较低但不为零。
- `q_annotation_agreement` 只来自真正独立的 original/reannotated agreement；无第二标注时不伪造。
- 项目已有证据支持这一必要性：paired reannotated train label Dice mean `0.862338`、Q10 `0.685225`，并存在空 mask 与面积膨胀 outliers。

### 2. Model uncertainty only weakly attenuates

OOF/augmentation uncertainty 不判断标签对错，只允许有上限的弱调制：

```text
u_model in [0,1]
q_x = q_label(x) * (1 - delta*u_model(x))
q_x = stopgrad(q_x)
0 <= delta <= delta_max, delta_max fixed globally (e.g. 0.2)
```

因此困难真边界最多再下调固定小比例，不会因 morphology、OOF、augmentation、annotation 四项连乘而失去监督。禁止用“teacher 与标签一致”提高权重。

### 3. Piecewise robust base loss

GCE 真正替代不确定边界上的 BCE，而不是叠加：

```text
L_pix = mean_{x in C_fg union C_bg}
        q_x * BCE(p_x,y_x)
      + mean_{x in U_bd}
        q_x * GCE_gamma(p_x,y_x)

L_wDice = 1 - (2 sum q_x p_x y_x + eps)
                  /(sum q_x p_x + sum q_x y_x + eps)

L_base = L_wDice + L_pix
```

前景、背景分别归一化后再求和，避免远背景数量支配梯度。`GCE_gamma=(1-p_y^gamma)/gamma`，仅用于 `U_bd`。

### 4. Prior-calibrated pair coefficient

冻结 HA-PDSP prior 只输出：

```text
k_ij = stopgrad(
       P(y_sep=1 | image context, prediction geometry)
       * c_epi(context only)
       * c_candidate
       * c_stability)
```

它不输出 seam map、corridor label 或 cut。异常 prediction geometry 可以提高 separation posterior，但不能降低 context-only epistemic support。

### 5. Binary-mask valid-pair rule

训练 pair 有效必须同时满足：

1. 两个 prediction-derived candidate 分别与两个不重叠的 eroded foreground supports 匹配；
2. candidate generation 不看 GT，GT match 只在候选产生后用于训练监督；
3. 两个 support 的 geometric between-region 中确实存在原 bone label 的 `y=0` 像素；
4. 推理时完全不需要上述 GT match。

如果 binary GT 已把两骨完全连成一个前景、没有任何 `y=0`，该 pair 必须跳过，不能声称可被本 loss 修复。

### 6. Reliable negative anchors and hard-negative mining

```text
N_ij = {x:
        y(x)=0,
        x in between-region(i,j),
        q_label(x) >= q_floor}
```

- 不要求 EMA/teacher 背景一致；否则当前桥接会被 teacher 重复并排除。
- `q_floor` 采用适中的 annotation-side floor，不要求远背景级高置信，避免细骨缝全部被过滤。
- 在 `N_ij` 内选择当前 `p_x` 最高的 top-k hard negatives；top-k 索引 stop-gradient。
- 当前预测只决定优先优化哪些“原标签已经为背景”的像素，不决定标签方向。

定义 `h_x` 为 stop-gradient top-k indicator/soft hard-negative weight：

```text
beta_x = 1 + lambda_pair
         * Agg_{ij:x in N_ij} k_ij*h_x

L_pair = - mean_{x:y_x=0}
         q_label(x) * (beta_x-1) * log(1-p_x)
```

这会直接降低相邻骨头之间已有可靠背景 hard negatives 的错误前景置信，而不能通过单纯提高 bone core confidence 逃避。

### 7. Final two-term loss and cross-model balancing

```text
g_b = ||dL_base/dl||_2
g_p = ||dL_pair/dl||_2

if no_valid_pair or g_p < tau_g:
    L = L_base
else:
    alpha = clip(
        rho * stopgrad(g_b/(g_p+eps)),
        alpha_min, alpha_max)
    L = L_base + alpha*L_pair
```

- norm 只对最终 mask logits `l` 计算；YOLO detection/objectness loss 不进入 ratio。
- `alpha` stop-gradient，不产生二阶梯度；实现可用 `autograd.grad(...,retain_graph=True)`。
- 同一 `q_min,delta_max,gamma,q_floor,top-k,rho,clip,warm-up` 在 original train/val 冻结，跨模型共享。
- warm-up 阶段只训练 `L_base`，candidate 稳定后启用 `L_pair`。

## Exact Capability Boundary

- 能修复：原标签中存在背景证据、但普通 loss 对其权重不足或模型把它预测成前景桥的情况。
- 能缓解：边界噪声、空/异常 label 对梯度的破坏以及不稳定边界过拟合。
- 不能修复：GT 和所有可用监督都把两骨完整连成一个前景且不存在任何负像素的情况；此时 confidence reweighting 没有合法负监督，只能 skip。

## Minimal Validation Sketch（当前不执行）

### Claim 1: Noise-robust base loss

- Native Dice+BCE vs piecewise BCE/GCE vs annotation reliability vs bounded uncertainty attenuation。
- Metrics: Dice/Recall/Boundary，label-disagreement subgroup，core/boundary gradient mass。

### Claim 2: Prior coefficient strengthens existing separation evidence

- No pair term vs uniform hard-negative reweight vs prior-calibrated `L_pair`。
- Metrics: valid-pair coverage、merge rate、gap FP、component MAE、harmful activation、Recall non-inferiority。

### Cross-model lock

- Same loss construction and frozen prior across at least 5 model families。
- clean-test-v2 only after original train/val freeze under R201。

