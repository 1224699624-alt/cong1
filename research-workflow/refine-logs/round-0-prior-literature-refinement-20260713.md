# Research Proposal: Heterochrony-Aware Prediction-Derived Separation Prior (HA-PDSP)

## Problem Anchor

- Bottom-line problem: 建立一个与基础分割网络解耦的儿童手部骨骺先验，使同一冻结先验能在多种分割模型上改善骨缝粘连、相邻骨误连接、边界质量和解剖一致性。
- Must-solve bottleneck: 儿童骨骺不仅随年龄和性别变化，同一个儿童的不同区域也可能异步成熟；固定形状、固定组件数或单一全局成熟标量会把正常差异当成错误。
- Non-goals: 不设计新 backbone；不生成完整 foreground atlas；不鼓励骨头前景连通；不依赖固定实例 ID；不使用 clean-test-v2 调参；不直接声称建模完整指骨长度。
- Constraints: 仅 `TSRS_RSNA-Epiphysis`；严格 R201；不改变原始数据结构；先验、候选算法和集成规则跨模型冻结；仅 original train/val 建模和选择。
- Success condition: 同一冻结先验在至少 4/5 个异构模型上改善 Boundary IoU/F1、Surface Dice、HD95/ASSD、gap FP、merge rate 和 component-count MAE，同时 Dice/IoU/Recall 非劣。

## Technical Gap

现有显式 shape prior、SSM 和 deformation prior 多假设单一器官、固定组件数或固定 topology；Deep Closing 和 reconnecting regularizer 默认前景应连通；Fast χ 需要预先知道正确 Euler 状态；SPM 的 dense global/local maps 又会把先验变成第二分割器。儿童骨骺真正缺少的是一个能够表达“全局成熟 + 局部异步成熟 + 关系方差”，但不独立预测空间 mask 的条件关系先验。

## Method Thesis

学习一个冻结的、发育条件化的骨骺关系分布，而不是平均手形：X 光只估计全局和粗区域成熟 posterior，任意基础模型的 soft mask 经统一算法产生候选图，先验通过方差校准的 should-separate 概率和 max-min 前景桥连通度，只惩罚高置信度错误粘连。

## Contribution Focus

- Dominant contribution: heterochrony-aware developmental relation prior，在不固定组件数和实例身份的条件下建模局部骨缝关系，并以同一输出空间能量接入不同模型。
- Supporting contribution: masked graph corruption 让先验与 backbone 解耦地预训练，并将能量梯度解释为局部 topology violation map。
- Explicit non-contributions: 不提出 dense shape generator、deformation atlas、第二 correction network、通用骨龄模型或完整手骨 landmark 系统。

## Proposed Method

### 1. Deterministic hand coordinate system

统一左右手方向、hand ROI、旋转和长宽尺度。只使用 hand-normalized 坐标。固定划分 `3×3` 粗区域：radial/central/ulnar × proximal/middle/distal。该区域仅用于 pooling 和统计分组，不声称对应具体命名骨骺。

### 2. Global-regional developmental posterior

冻结图像 encoder 的 dense feature 不输出 mask，只进行固定 pooling：

```text
z_g ~ q_g(GlobalPool(F(x)), age_if_available, sex)
z_r ~ q_r(RegionPool_r(F(x)), z_g), r=1..9
```

- `z_g` 表达整手成熟水平；`z_r` 是受零均值 shrinkage 约束的局部成熟残差，表达腕部、掌骨区和不同手区不同步。
- 每个 posterior 同时输出 mean 与 variance；区域证据弱或受曝光影响时 variance 增大。
- 无 spatial decoder、skip 或位置热图，最多输出 1 个全局 latent 和 9 个区域标量/低维 latent，因此不是粗分割器。
- chronological age 只在元数据真实存在时使用；RSNA `boneage` 明确为 radiographic bone age，只能在 original-train 作可删除 ordinal teacher；sex 是带 shrinkage 的弱协变量。
- 曝光、对比度、旋转和缩放增强前后要求 `z_g,z_r` 一致，避免把成像域当成成熟差异。

### 3. Prediction-derived candidate graph

prior pretraining、backbone training 和 inference 使用完全相同的 stop-gradient 管线：

```text
soft mask -> distance transform -> local peaks -> watershed
          -> normalized nodes -> Delaunay edges -> frozen length cap
```

节点默认无命名，只含区域、中心、面积、尺度、basin/peak confidence。缺少骨化中心时不生成虚拟节点；空图 abstain。当前标签只支持骨骺中心、面积、相对位置、间距与 gap width，不支持 shaft length。

### 4. Heteroscedastic developmental relation bank

对每条 edge 定义关系向量：

```text
e_ij = [relative displacement, log area ratio,
        normalized gap width, regional pair, candidate confidence]
```

关系先验同时输出：

```text
p_sep = P(two candidates are distinct GT components | z_g,z_ri,z_rj,e_ij)
p(e_ij | z_g,z_ri,z_rj) = sum_k pi_k N(mu_k, diag(sigma_k^2))
```

这里的 mixture/variance 取代单一平均 SSM：稳定关系方差小、约束强；发育差异大的关系方差大、自动降权。`K` 和所有统计规则只在 original train/val 冻结一次。

### 5. Masked graph corruption pretraining

所有训练输入都经过同一候选管线，并保留原始 GT component provenance：

- bridge corruption：连接不同 GT components，标签 `should-separate=1`；
- over-split corruption：分裂同一 component，标签 `should-separate=0`；
- component dropout：模拟尚未骨化、弱显示或模型漏检，但不要求恢复虚拟 mask；
- position/scale jitter：模拟姿态归一化误差和个体几何变化；
- masked edge attributes：预测被遮蔽的关系分布，学习 backbone-independent relational context。

```text
L_prior = BCE(p_sep,y_sep)
        + lambda_rel * NLL_mixture(e_ij)
        + lambda_cal * calibration_loss
        + optional lambda_ord * radiographic_age_teacher
```

不同 provenance ID 为正例，同一 ID 为 over-split 负例；背景、混合或不唯一候选 ignore；non-neighbor 不当负例。

### 6. Selective relation gate

借鉴 SSM adaptive fusion 的“只在先验更可靠时介入”，但不融合两个 dense masks：

```text
w_ij = p_sep
     * relation_likelihood(e_ij)
     * inverse_posterior_uncertainty
     * candidate_stability
```

只有 `w_ij` 超过冻结阈值、候选在轻微增强下稳定、区域 maturity posterior 方差低时才激活。罕见发育模式、弱骨化中心和 OOD 影像自动 abstain。

### 7. Reversed-topology separation energy

在每条激活 edge 的固定 `64×64` crop 上计算 strongest max-min foreground path：

```text
r_0 = stopgrad(source_seed)
r_{t+1} = max(r_0, min(p_crop, MaxPool3x3(r_t)))
T = 64
Conn_ij = mean(r_T[target_seed])

E_sep = sum(w_ij * Conn_ij) / (eps + sum(w_ij))
```

它只惩罚两个本应分开的候选之间存在高置信前景桥，不要求所有前景连通，也不会把微弱背景响应反复累积。

### 8. Local violation map without a correction network

借鉴 Fast χ 的 violation localization，但不使用固定 Euler 目标：

```text
V = abs(dE_sep / dl)
```

`V` 是 max-min bridge energy 对最终 logits 的梯度，只突出决定最强错误连接的瓶颈路径。它用于三件事：限制 prior gradient 的空间支持、生成可审计热图、统计 prior 实际干预面积。它不是新的 trainable module。

### 9. Cross-model integration

```text
g_n = ||dL_native/dl||_2
g_p = ||dE_sep/dl||_2

if no_edge or g_p < tau_g:
    L = L_native
else:
    alpha = clip(rho * stopgrad(g_n/(g_p+eps)), alpha_min, alpha_max)
    L = L_native + alpha * E_sep
```

prior encoder、relation bank、candidate graph、threshold、gradient ratio、warm-up 和 abstention 规则全部冻结并跨 nnU-Net、ARAA、YOLO+SAM trainable mask head/refiner、DINOv3/U-Net 等模型共用。

## Why the Six Papers Improve but Do Not Replace the Method

- 从 SPM 吸收 global/local 双层思想，但把 local dense shape map 改成 9 个 coarse maturity residuals。
- 从 SSM adaptive fusion 吸收 variance weighting 和 selective gating，但不生成/融合平均形状 mask。
- 从 Deep Closing 吸收 masked corruption 与 topology-critical focus，但反转为背景骨缝分离，不做前景 reconnect。
- 从 plug-and-play reconnecting regularizer 吸收独立预训练和冻结复用，但使用 GT-derived bone corruptions 而非通用管线生成器。
- 从 TEDS-Net 吸收 topology/geometry 分离，但拒绝固定 topology deformation prior。
- 从 Fast χ 吸收 violation map 思想，但 violation 来自发育条件 max-min bridge energy，不假定固定 Euler characteristic，也不增加 correction network。

## Failure Modes and Diagnostics

- Regional latent 仍不足以表达异步成熟：检查 `z_r=0` 与 global+regional 的增益；若无增益删除 regional head。
- Candidate pipeline 漏掉弱中心：报告 candidate coverage 和 abstention，不用第二定位器补齐。
- Mixture relation bank 过拟合：比较 K=1 与小 K mixture；按 train/val NLL 和 calibration 冻结，不按 test 选择。
- Prior 错切真实骨头：监控 over-split negative、Recall、cut-region GT foreground fraction 和 violation-map intervention area。
- Acquisition domain shift：按曝光/分辨率/OOD score 分层报告 prior activation 和失败率。

## Claim-Driven Validation Sketch（当前不执行）

### Claim 1: 发育异步条件关系先验可跨模型降低误连接

- Baselines: no prior；固定 seam atlas；age/sex hard bins；上一版 global-only PD-DSP；HA-PDSP。
- Metrics: merge rate、gap FP、component-count MAE、Boundary IoU/F1、Surface Dice、HD95/ASSD；Dice/IoU/Recall 非劣。
- Evidence: 同一冻结 prior 在至少 4/5 backbone 上方向一致。

### Claim 2: 局部 heterochrony、variance gating 和 masked corruption 是必要机制

- Ablations: 去 `z_r`；去 mixture variance；去 corruption；去 selective gate；用固定 Euler/前景 connectivity 替代。
- Evidence: 低龄、组件少、局部成熟不一致和低质量 X 光子组中，完整方案降低 harmful activation 与 Recall 损失。

### Safety/Protocol block

- 全部开发限 original train/val，按年龄/性别/组件数/图像质量分层。
- clean-test-v2 在 prior 和所有规则冻结后一次性 R201 final evaluation。
- 不混入 Articular-Surface，不改变原始数据结构。

