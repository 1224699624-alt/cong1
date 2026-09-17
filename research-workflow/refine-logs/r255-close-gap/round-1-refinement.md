# Round 1 Refinement: Development-Conditioned Continuous Bridge Risk

## Problem Anchor

- **Bottom-line problem**: 儿童手部 X 光骨骺分割中，主要粘连发生在腕骨/掌骨近端簇。低骨龄儿童骨化中心少、骨间距大，通常不需要扩大前景或额外干预；高骨龄儿童骨化中心逐渐发育、尺寸增大、局部骨缝缩窄，尤其是骨边界局部逼近时容易被分割模型连成同一前景组件。
- **Must-solve bottleneck**: 在不切掉真实骨头、不凭空生成骨缝标签、不让低骨龄病例发生错误碎裂或小骨化中心消失的前提下，降低高骨龄近距离骨对最窄瓶颈处的前景置信，阻断假连接。
- **Non-goals**: 不追求全手组件数优化；不使用推理期骨龄切割器；不做通用前景收缩；不生成 dense seam mask；不以 Dice SOTA 为唯一目标。
- **Constraints**: 只做 TSRS_RSNA-Epiphysis；原数据结构不修改；anchor 必须来自原 bone mask 中的 y=0；无可靠 y=0 则跳过；clean-test-v2 锁定；首版 nnU-Net。
- **Success condition**: original-val 腕骨/掌骨近端 1–4 px close-gap FP 和 false merge 明显下降，Precision、Boundary IoU/F1 不下降，Recall/Dice 和小骨化中心 Recall 非明显退化；低骨龄 false split 不增加。

## Anchor Check

- 原始瓶颈仍是高骨龄近距离腕骨簇假连接。
- 删除跨病例骨编号不会改变问题，只删除了不可靠假设。
- 骨龄仅作训练期 privileged signal，不改变标准 nnU-Net 推理。
- component count 仅为 guardrail，不再作为优化目标。

## Simplicity Check

- 唯一贡献：发育条件化的连续桥接风险用于训练期可靠负梯度分配。
- 删除：semantic pair ID、presence head、rank loss、推理期发育估计器、core BCE、hard same-component threshold。
- 新增参数网络：0。
- 推理额外成本：0。

## Revised Method

### 1. Canonical base

```text
L_base = canonical nnU-Net deep-supervision Dice + CE
```

close-gap 项只加在 full-resolution logits，不进入低分辨率 deep supervision，避免 1–2 px 缝隙下采样消失。

### 2. Training-only local instance sidecar

原始实例值标签作为隔离 sidecar，与 binary target 同步执行同一几何增强。它只在单张图内标识不同连通骨支持，不要求跨病例语义 ID。

如果某标签内部已经粘连、断裂异常或增强后实例不可稳定恢复，则该 pair abstain。

### 3. Orientation-normalized local descriptor

每个图内邻近 pair 使用无身份描述符：

```text
z_ij = [
  normalized pair midpoint,
  relative boundary direction,
  boundary distance / local bone width,
  log area ratio,
  compactness pair,
  local ROI coordinate
]
```

手部主轴由骨支持 PCA 估计，方向通过远端指骨密度确定；无法稳定定向时只使用旋转不变量并降低置信。腕骨/掌骨近端 ROI 在训练前按 orientation-normalized hand support bbox 固定，不从验证改善病例手工选择。

### 4. Developmental conditional closeness

用训练病例按连续 bone age 和 sex 对局部描述符距离分布做核平滑：

```text
r_dev(i,j) =
  support_ESS(a,s,z_ij)
  × P(pair is unusually close | a,s,z_ij)
```

每病例先归一化 pair 权重，避免高骨龄因骨化中心多而天然贡献更多 loss。

bone age 只在训练时使用：

```text
bone age -> stop-gradient training coefficient
inference -> standard f_theta(X)
```

低骨龄骨化中心少且距离大时，pair 数少、r_dev 低，自然 abstain；不引入显式前景扩张。

### 5. Threshold-free continuous bridge risk

从 detached full-resolution foreground probability p 构建两个支持之间的 widest-path score：

```text
b_ij = max over paths P(i->j) of min_{x in P} p(x)
```

b_ij 表示连接两个骨支持所需路径的最弱前景概率。它连续地刻画当前预测是否形成高置信桥，不使用 0.5 连通阈值，也不反向传播通过路径选择。

pair coefficient：

```text
k_ij = stopgrad(r_dev × b_ij × q_xray × q_aug × q_ann)
```

### 6. Exact X-ray separation evidence

沿两个实例最近边界点连线采样局部强度剖面。用两侧骨内强度和局部标准差归一化：

```text
q_valley = clip(
  [min(I_bone_i, I_bone_j) - I_gap]
  / [abs(I_bone_i-I_bone_j) + sigma_local + eps],
  0, 1
)

q_xray = q_valley × q_double_edge
```

q_double_edge 要求谷两侧存在方向相反、分别指向两个骨支持的梯度。缺少谷或双边缘时 abstain。灰度极性由局部骨内/骨间相对强度决定，不使用全局固定阈值。

### 7. Narrow-gap reliability

候选 anchor 必须：

- 位于原始/增强后 full-resolution binary target 的 y=0；
- 位于最近边界 bottleneck corridor；
- 增强后至少保留一个背景像素；
- 在两次轻微 photometric/geometric view 下位置和 q_xray 稳定；
- 对 1 px gap，只有 q_xray 和 q_aug 都超过训练预注册门槛才保留。

记录 augmentation 前后 gap 消失率、拓扑改变率和 anchor coverage。无合法 anchor 则跳过。

### 8. Single robust pair loss

首版删除 rank loss，仅使用 Huberized/capped background penalty：

```text
ell_bg(z) = c * tanh(softplus(z_fg-bg) / c)

L_close = sum_{images} [
  sum_x max_pair(k_ij q_x ell_bg(x))
  / (eps + sum_x max_pair(k_ij q_x))
]

L = L_base + alpha L_close
```

- max 聚合避免 pair 重叠放大；
- 每图归一化后再 batch 平均，避免高骨龄 pair 数更多；
- alpha 由 full-resolution logits 梯度比限制在预注册范围；
- gap probability 已低于停止门槛时不再施压；
- 每图 anchor/pair 数设上限。

### 9. Warm-up and guards

- epoch 0–2：仅 native nnU-Net；
- 之后启用 detached bridge-risk；
- 监控 foreground area loss、每实例 Recall、小中心 Recall、false split、fragment increase、Boundary IoU/F1；
- 所有阈值在 train-only audit 预注册。

## Minimal Validation

### Gate A — train-only anchor audit

无需训练，报告：

- 原始与预处理 gap width；
- gap/local bone width；
- 按 bone-age 连续区间的 pair count 与 ESS；
- 1–2/3–4/5–8/>8 px anchor coverage；
- augmentation survival；
- q_xray 分布；
- low-age activation 应低但不能靠硬年龄阈值。

### Gate B — 2-epoch sanity

比较 native vs close-gap loss：

- selected bottleneck p(foreground) 是否下降；
- bone core/small-center Recall 是否稳定；
- false split 和 fragment 是否不增加；
- 梯度冲突是否受限。

### Gate C — original-val pilot

决定性指标：

- wrist/carpal-metacarpal close-gap FP by width；
- false merge rate；
- Precision、Boundary IoU/F1；
- Dice/Recall/small-center Recall guardrails。

只有 Gate C 通过才讨论 age-conditioning ablation；未通过不得运行 clean-test-v2。

