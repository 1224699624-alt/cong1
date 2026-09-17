# Final Proposal: HA-PDSP + PCR-Loss

## 方法转向

上一版 HA-PDSP 的 max-min bridge energy 不再作为训练目标。发育先验现在只输出 stop-gradient pair coefficient；训练方向完全由原始 bone mask 决定。PCR-Loss 只加强原标签已有的可靠背景 hard negatives，并对噪声边界保守降权，不构造反向骨缝标签。

## 1. Noisy-label base loss

从 bone mask 定义高可信前景 core、高可信远背景和不确定边界带：

```text
C_fg = erosion(y,r_core)
C_bg = outside(dilation(y,r_far))
U_bd = remaining pixels
```

```text
q_label = q_min + (1-q_min)*q_ann_morph
q_x = stopgrad(q_label*(1-delta*u_model))
```

- `q_ann_morph` 来自 morphology 和真实存在的独立标注一致性；
- OOF/augmentation uncertainty 只允许有上限的弱衰减，不能判断标签对错；
- 禁止用 teacher-label agreement 提高权重；
- `q_min` 保证困难真边界仍有监督。

```text
L_pix = separately_normalized BCE on C_fg/C_bg
      + GCE on U_bd

L_base = L_weightedDice + L_pix
```

BCE 与 GCE 区域互斥，不使用额外 teacher consistency。

## 2. Prior only changes pair coefficients

冻结 HA-PDSP 输出：

```text
k_ij = stopgrad(
       P(two candidates should be distinct | image context,geometry)
       * context-only epistemic confidence
       * candidate confidence
       * stability)
```

prior 不输出 seam、corridor mask、background target 或 cut。

## 3. Legal negative anchors

训练 candidate 先由模型 prediction 产生，之后才能匹配两个不重叠的 eroded GT foreground supports。只从原 bone mask 已经为背景的像素选择：

```text
N_ij = {x: y(x)=0,
           x in between-region(i,j),
           q_label(x)>=q_floor}
```

如果 binary GT 将两骨完整标成连通前景、between-region 没有任何 `y=0`，该 pair 严格跳过。between-region 只是搜索范围，不是 seam label。

## 4. Prior-calibrated hard-negative coefficient

在 `N_ij` 内按当前 `p_x` 选择 top-k foreground hard negatives，索引 stop-gradient：

```text
a_x = max_{ij:x in N_ij} [k_ij*h_ij(x)]
0 <= a_x <= 1
```

使用 `max` 防止多个 pair 重复放大同一像素。仅按实际有效负锚点权重归一化：

```text
Z_pair = sum_{x:y=0} q_label(x)*a_x

if Z_pair <= eps:
    L_pair = 0
else:
    L_pair = -sum q_label*a_x*log(1-p)/(eps+Z_pair)
```

target 方向始终来自原 bone mask 的 `y=0`；prediction 只决定优化优先级；prior 只决定梯度系数。

## 5. Cross-model loss

```text
g_b = ||dL_base/dl||_2
g_p = ||dL_pair/dl||_2

if Z_pair<=eps or g_p<tau_g:
    L = L_base
else:
    alpha = clip(rho*stopgrad(g_b/(g_p+eps)),
                 alpha_min,alpha_max)
    L = L_base + alpha*L_pair
```

norm 只对最终 mask logits 计算；YOLO detection/objectness loss 不进入 ratio。所有 reliability、top-k、prior、gradient-ratio 和 warm-up 规则仅在 original train/val 冻结并跨模型共享。

## 能力边界

- 能改善：标签已有细骨缝背景，但模型把它预测成前景桥；噪声边界对普通 BCE/Dice 产生过强错误梯度。
- 不能改善：标签把两骨完整标成一个前景且完全没有背景负像素。此时 confidence reweighting 没有合法监督，必须 skip。
- 这不是反向背景拓扑监督，而是 bone-mask supervision 下的可靠性与 hard-negative gradient allocation。

## 验证边界（当前不执行）

- Native Dice/BCE、piecewise robust base、reliability base、full PCR-Loss。
- Uniform hard-negative weight vs prior-calibrated coefficient。
- 报告 valid-pair coverage、selected anchors、merge/gap/component metrics、Boundary metrics、Recall/Dice 非劣和 harmful activation。
- 同一规则接入至少 5 个模型；clean-test-v2 只在完全冻结后作一次 R201 final evaluation。

