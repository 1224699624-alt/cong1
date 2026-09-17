# Round 2 Refinement: HA-PDSP

## Problem Anchor

- Bottom-line problem: 建立一个与基础分割网络解耦的儿童手部骨骺先验，使同一冻结先验能在多种分割模型上改善骨缝粘连、相邻骨误连接、边界质量和解剖一致性。
- Must-solve bottleneck: 儿童骨骺不仅随年龄和性别变化，同一个儿童的不同区域也可能异步成熟；固定形状、固定组件数或单一平均模板会把正常差异当成错误。
- Non-goals: 不设计新 backbone；不生成完整 foreground atlas；不鼓励骨头前景连通；不依赖固定实例 ID；不使用 clean-test-v2 调参；不直接声称建模完整指骨长度。
- Constraints: 仅 `TSRS_RSNA-Epiphysis`；严格 R201；不改变原始数据结构；先验、候选算法和集成规则跨模型冻结；仅 original train/val 建模和选择。
- Success condition: 同一冻结先验在至少 4/5 个异构模型上改善 Boundary IoU/F1、Surface Dice、HD95/ASSD、gap FP、merge rate 和 component-count MAE，同时 Dice/IoU/Recall 非劣。

## Anchor and Simplicity Check

- 核心仍是一个统一的 developmental relation posterior + 一个固定的 max-min bridge energy。
- 本轮不新增模块，只把 image-only conditioner、prediction-derived geometry 和 gate reliability 的概率角色彻底分开。
- local feature 只称 candidate-conditioned radiographic/ossification evidence，不声称是可解释区域骨龄。

## Finalized Proposal

### 1. Inputs and frozen prior encoder

推理时只使用 X 光、真实可用的 sex/chronological age，以及基础模型自己的 soft prediction。hand ROI、左右手统一、坐标归一化、candidate seed 均禁止使用 GT。

```text
F = Encoder(x)
z_g = GlobalPosterior(GlobalPool(F), age_if_available, sex)
a_i = PatchPool(F, stopgrad(candidate_center_i), frozen_radius_rule)
```

- `z_g` 是全局发育 posterior；`a_i` 是候选位置触发的局部影像/骨化证据。
- encoder 可在 prior pretraining 阶段训练一次；prior 完成后与 relation model 一起冻结，所有目标 backbone 共享。
- 无 decoder、dense prior map、heatmap 或节点生成；没有候选就没有 `a_i`。
- RSNA radiographic `boneage` 只可作为 original-train 可删除 teacher，不是部署输入，也不能改称 chronological age。

### 2. Shared candidate graph

所有 synthetic、OOF、backbone training 和 inference soft masks 使用同一 stop-gradient pipeline：probability normalization、distance transform、peak、watershed、hand-normalized node、Delaunay 和冻结长边上限。节点无固定实例身份；缺少骨化中心时不创建虚拟节点；空图 abstain。

### 3. Correct relation factorization

将 image-only/context conditioner 与需要评价的 prediction geometry 分开：

```text
c_ij = [z_g, a_i, a_j,
        coarse image coordinates,
        sex/chronological_age_if_available]

u_ij = [relative center displacement,
        log basin-area ratio,
        normalized basin scales]
```

`a_i,a_j` 和坐标只作为 conditioner，不对其自身做 Gaussian NLL。定义唯一的 class-conditional model：

```text
p(u_ij | y_sep, c_ij)
  = N(mu_y(c_ij), diag(max(sigma_y(c_ij), sigma_min)^2))

P(y_sep | u_ij,c_ij)
  ∝ p(u_ij | y_sep,c_ij) P(y_sep | c_ij)
```

- 默认单 Gaussian；只有统一 original-val calibration/NLL 明确支持时才整体升级为小 K mixture，不能按 backbone 选择。
- `sigma_y` 描述给定儿童发育和局部影像条件后的 aleatoric 几何变异；高变异关系自然产生较软 posterior。
- posterior 与 relation density 是同一 Bayes mechanism，没有独立 classifier/NLL 双头。

### 4. Training data without candidate leakage

- Synthetic bridge：不同 GT provenance，`y_sep=1`。
- Synthetic over-split：同一 GT provenance，`y_sep=0`。
- Small geometry/appearance jitter：鲁棒性增强。
- Patient-disjoint original-train OOF soft predictions：candidate 只从 prediction 产生，GT 仅在 edge 已产生后赋 provenance label。
- 背景/混合/不唯一 edge ignore；non-neighbor 不当负例。
- 删除 masked reconstruction、component dropout 和 presence/count 支线。

```text
L_prior = -log p(u_ij | y_sep,c_ij)
        - log P(y_sep | c_ij)
        + lambda_cal * posterior_calibration
        + lambda_inv * augmentation_consistency
        + optional lambda_ord * radiographic_age_teacher
```

### 5. Reliability gate cannot veto abnormal geometry

epistemic/OOD confidence 严禁读取当前 mask-derived geometry `u_ij`、gap、Conn 或 bridge strength：

```text
c_context = [z_g, a_i, a_j,
             coarse image coordinates,
             acquisition/image-quality representation]

w_ij = P(y_sep=1 | u_ij,c_ij)
     * c_epi(c_context)
     * c_candidate
     * c_stability
```

- `c_epi` 只回答“这种儿童/局部影像条件是否有训练支持”，不回答“当前预测几何是否正常”。
- 异常 `u_ij` 是 separation posterior 的修复证据，不能成为 abstention 证据。
- `c_candidate` 单独衡量是否存在两个可靠 peak/basin；`c_stability` 衡量候选在轻微扰动下是否稳定。
- aleatoric `sigma_y` 只影响 posterior softening，不充当 epistemic veto。

### 6. Reversed-topology energy

```text
r_0 = stopgrad(source_seed)
r_{t+1} = max(r_0, min(p_crop, MaxPool3x3(r_t)))
T = 64
Conn_ij = mean(r_T[target_seed])

E_sep = sum(w_ij * Conn_ij) / (eps + sum(w_ij))
```

该 strongest max-min path 只惩罚两个应分离候选之间的高置信前景桥，不鼓励骨头前景连通。

### 7. Cross-model integration

```text
g_n = ||dL_native/dl||_2
g_p = ||dE_sep/dl||_2

if no_edge or g_p < tau_g:
    L = L_native
else:
    alpha = clip(rho * stopgrad(g_n/(g_p+eps)), alpha_min, alpha_max)
    L = L_native + alpha * E_sep
```

prior、candidate rules、calibration、gate 和 gradient-ratio 在 original train/val 冻结，跨所有目标模型共享。

### 8. Audit-only violation map

```text
V = stopgrad(abs(dE_sep/dl))
```

只用于热图、intervention area 和案例审计；不反馈 loss、不 mask gradient、不参与更新。

## Literature-grounded method boundary

- 继承 plug-and-play prior 的独立预训练/冻结接口；
- 继承 SSM 的 variance-aware relation，但不使用平均 shape mask；
- 继承 SPM 的 global/local 分工，但 local 仅在已有候选位置池化；
- 继承 Deep Closing 的 topology corruption，但反转为背景骨缝 separation；
- 继承 TEDS 的 topology/geometry 分离，但不固定 topology；
- 继承 Fast χ 的 violation visualization，但不设固定 Euler target或 correction network。

## Minimal validation boundary（当前不执行）

- 5 个异构模型，同一冻结 prior；报告结构指标和 Dice/IoU/Recall 非劣。
- 报告 candidate coverage、activation、abstention、harmful activation。
- synthetic-only vs synthetic+OOF；global-only vs global+candidate local evidence；无 epistemic gate。
- leave-one-backbone-out 检验 unseen backbone transfer。
- 所有开发仅 original train/val；clean-test-v2 只作冻结后的 R201 final evaluation。

