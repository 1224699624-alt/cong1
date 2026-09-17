# Round 2 Refinement

## Problem Anchor

- Bottom-line problem: 建立一个与基础分割网络解耦的儿童手部解剖先验，使同一先验在 nnU-Net、ARAA、YOLO+SAM、DINOv3/U-Net 等不同模型上都能改善骨缝粘连、边界质量和解剖一致性。
- Must-solve bottleneck: 儿童骨骺随年龄、性别和个体成熟度高度异步变化，固定模板会把正常发育差异误判为分割错误。
- Non-goals: 不设计新 backbone；不预测完整 foreground atlas；不使用 clean-test-v2 调先验；不鼓励骨头前景连通；不假设实例 ID 固定。
- Constraints: 仅 TSRS_RSNA-Epiphysis；R201；原始数据不变；先验和集成规则跨模型冻结；必须先回连并审计 RSNA metadata。
- Success condition: 同一冻结先验在至少 4/5 个异构模型上改善骨缝、边界和组件指标，同时 Dice/IoU/Recall 非劣。

## Anchor Check

- 原始瓶颈仍是儿童发育变化下的相邻骨误连接，而不是建立完整手骨生成模型。
- 本轮删除先验自身的空间定位能力；它只在基础模型输出上判断局部组件关系，因此更贴合“模型无关先验”。
- 不引入第二分割器、完整指骨 landmark 网络或模型专属 adapter。

## Simplicity Check

- 唯一主贡献：发育条件化、预测派生的负空间关系能量。
- 冻结复用：一个全局成熟度 encoder；同一候选构图算法；同一关系 scorer；同一输出空间归一化规则。
- 删除：学习型 pose latent、先验生成的节点位置/支持区、named anatomy 强依赖、完整指骨长度、多个可替换 topology operator。

## Revised Proposal: Prediction-Derived Developmental Separation Prior (PD-DSP)

### 1. 一句话方法

从 X 光只估计低维全局发育状态，从任意基础模型的 soft mask 用固定算法提取局部候选图，再以训练集学得的发育条件关系概率惩罚“不应连接的候选被前景桥接”；先验不独立预测任何样本级骨骺位置或 mask。

### 2. 输入与发育状态

部署形式为：

```text
q(z_dev | GlobalPool(g(x)), sex, chronological_age_if_available)
```

- `g` 无 spatial decoder、skip 或 dense output；`z_dev` 为标量或至多 4 维 posterior。
- 标准 RSNA metadata 中的 `boneage` 必须标记为 radiographic bone age，不得改称 chronological age。
- radiographic bone age 只可作为 original-train 上的可选 ordinal teacher；测试时删除 teacher head。
- 若无真实 chronological age，部署式退化为 `q(z_dev | x, sex)`，不从图像“推断 chronological age”。
- sex 仅作为带 shrinkage 的弱协变量，不能形成男女硬模板。
- 使用左右手统一、确定性 hand ROI、旋转/尺度归一化；增强一致性使 `z_dev` 对曝光、对比度、旋转和缩放不敏感。先验不学习 `z_pose`。

`z_dev` 的主监督来自 original-train GT 图的低维发育描述（组件数、骨骺面积分布、中心间距分布），radiographic bone age 仅是可删除的辅助排序监督。

### 3. 固定的 prediction-derived candidate graph

对任意模型归一化后的 soft foreground probability `p` 执行同一算法：

1. 对 `stopgrad(p)` 做距离变换和局部极大值检测；
2. 用固定 watershed 得到候选区域与中心；即使两个骨骺被细桥粘连，两个距离峰仍可形成两个候选；
3. 在 hand-centric 相对坐标中建立 Delaunay 图，并删除超过训练集尺度上限的边；
4. 节点仅含区域坐标、面积、局部尺度和置信度；默认不赋予具体指骨名称。

图身份采用可降级层级：

```text
named family/ray（仅经额外审计证明可靠时）
  -> radial/central/ulnar × proximal/middle/distal region
  -> unlabeled local Delaunay node（保证可用的默认层）
```

现有 epiphysis mask 只支持骨骺中心 presence、面积、相对位置、间距与 gap width；不声称建模完整指骨长度、shaft width 或关节轴。

### 4. 发育条件关系先验

先验 scorer 仅对 prediction-derived edge 输出：

```text
c_ij = P(two candidate regions should be separated |
         z_dev, sex, regional coordinates,
         relative spacing, relative scale, candidate confidence)
```

训练样本仅由 original-train GT 构造：

- 正例：两个相邻 GT 组件之间应保持背景分隔；
- bridge corruption：用细桥合并相邻 GT 组件，仍标记为“应分开”；
- over-split corruption：把同一 GT 组件人工分裂，标记为“不应强制分开”；
- non-neighbor pair：作为低权重负例。

这种监督直接区分真实骨缝与错误过分割，而无需稳定实例 ID。对于可可靠识别的区域组，可额外学习随 `z_dev` 变化的 count/presence intensity；对不可识别节点只使用 maturity-conditioned point-process count/intensity，不伪造 named presence curve。

### 5. 唯一的可微分 separation energy

本方案固定采用局部 fuzzy geodesic foreground connectivity，不再保留 soft-min、min-cut、Euler 等备选实现。

对 Delaunay edge `(i,j)` 的局部 crop：

```text
r_0 = stopgrad(seed_i)
r_{t+1} = SmoothMaxPool(min(r_t, p_crop))
Conn_ij = mean(r_T over stopgrad(seed_j))
E_sep = sum(c_ij * u_ij * Conn_ij) / (eps + sum(c_ij * u_ij))
```

其中 `seed_i/seed_j` 来自固定 watershed，`u_ij` 是发育 posterior、图关系和图像增强一致性共同给出的置信度。梯度只通过 `p_crop`，不通过候选提取或先验 scorer。该能量仅惩罚连接两个候选的前景路径；低置信度关系自动 abstain，不会给低龄儿童创造不存在的骨骺或骨缝。

### 6. 跨模型统一接口

每个基础模型只需提供归一化 logits `l` 与 `p=sigmoid(l)`。先验、构图阈值、图尺度、warm-up、置信阈值全部在 original train/val 冻结。

使用输出空间梯度归一化，而非标量 loss EMA：

```text
E_bar = E_sep / stopgrad(||dE_sep/dl||_2 + eps)
L = L_native + lambda * E_bar
```

因此主张是“同一冻结先验、同一 prediction-derived 构图算法、同一输出空间归一化规则”，不主张不同 backbone 具有完全相同的原始优化动力学。

### 7. 年龄、性别和异步发育如何被表达

- 年龄不是硬分箱模板，而是 `z_dev` posterior 的一个可选观测量。
- 性别只平移/校准成熟分布，不决定某个骨骺必然存在。
- X 光提供个体级成熟证据，解决同龄同性别仍发育不一致的问题。
- posterior variance 表达同一年龄/性别下的多模态或不确定成熟状态；高不确定样本降低 prior 权重。
- 组件数变化通过 maturity-conditioned count/intensity 与候选图共同表达，而不是固定 slot 数。
- acquisition/pose 变化由确定性归一化和增强不变性处理，不与发育 latent 混合。

### 8. 最小验证边界（当前不执行）

核心验证只需三块：

1. 跨模型：冻结 PD-DSP 接入 5 个异构模型，检查至少 4/5 的 gap FP、merge rate、Boundary IoU/F1、Surface Dice、HD95/ASSD，并对 Dice/IoU/Recall 做非劣检验。
2. 必要性：fixed seam atlas、age/sex hard bins、无 `z_dev`、无 uncertainty 与完整 PD-DSP 对比。
3. 安全性：按成熟度、性别、组件数和 prior abstention 分层，验证低龄/少中心/罕见发育样本不被强制切割。

所有阈值和模型选择只使用 original train/val；clean-test-v2 仅在协议冻结后一次性最终评估。

### 9. 剩余风险

- watershed 峰在弱小骨化中心上不稳定：以增强一致性和候选置信度 abstain，不增加定位网络。
- binary epiphysis mask 的组件可能不等于单一解剖中心：先做标签语义审计；无法确认时只保留 unlabeled/regional relation prior。
- `z_dev` 可能学习拍摄域：用 acquisition augmentation consistency 和按设备/强度统计的 OOD 审计约束。
- 全局 `z_dev` 是否提供超越 mask 自身几何的信息必须通过“无 z”删除实验验证；若无增益，应删除 image encoder，保留更简单的 prediction-derived graph prior。
