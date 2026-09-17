# Round 3 Refinement

## Problem Anchor

- Bottom-line problem: 建立一个与基础分割网络解耦的儿童手部解剖先验，使同一先验在 nnU-Net、ARAA、YOLO+SAM、DINOv3/U-Net 等不同模型上都能改善骨缝粘连、边界质量和解剖一致性。
- Must-solve bottleneck: 儿童骨骺随年龄、性别和个体成熟度高度异步变化，固定模板会把正常发育差异误判为分割错误。
- Non-goals: 不设计新 backbone；不预测完整 foreground atlas；不使用 clean-test-v2 调先验；不鼓励骨头前景连通；不假设实例 ID 固定。
- Constraints: 仅 TSRS_RSNA-Epiphysis；R201；原始数据不变；先验和集成规则跨模型冻结；必须先回连并审计 RSNA metadata。
- Success condition: 同一冻结先验在至少 4/5 个异构模型上改善骨缝、边界和组件指标，同时 Dice/IoU/Recall 非劣。

## Anchor and Simplicity Check

- 主贡献保持为一个发育条件化、prediction-derived 的负空间关系能量。
- X 光分支只输出全局成熟 posterior；局部节点、边和 seed 均来自基础模型输出的固定处理。
- 不使用 named instance、foreground atlas、learned pose、shaft/phalange length 或第二 correction network。
- 本轮只封闭监督标签、候选一致性、连通算子和跨模型梯度标定四个实现缺口。

## Finalized Method: Prediction-Derived Developmental Separation Prior (PD-DSP)

### 1. Developmental posterior

```text
q(z_dev | GlobalPool(g(x)), sex, chronological_age_if_truly_available)
```

`z_dev` 是标量或至多 4 维 posterior，无 dense decoder。RSNA `boneage` 明确视为 radiographic bone age，只能在 original-train 作为可选 ordinal teacher；若无真实 chronological age，绝不伪称存在。sex 是带 shrinkage 的弱协变量。确定性 ROI/左右手/旋转/尺度归一化与增强一致性用于剥离拍摄差异。

### 2. One shared candidate pipeline

训练 prior、训练 backbone 和推理时完全复用同一管线：

```text
soft mask
 -> stop-gradient distance transform
 -> fixed local-maximum seeds
 -> fixed watershed
 -> hand-normalized candidate centers
 -> Delaunay edges with a frozen train-set distance cap
```

所有 clean GT、bridge corruption、over-split corruption 和模型预测都先转换成 soft mask，再走该管线。不存在“训练按 GT component 建图、部署按 prediction 建图”的接口差异。

节点默认是 unlabeled local/regional node。只有独立标签审计证明可靠时才上升到 named family/ray。当前 epiphysis mask 只支持中心 presence、面积、相对位置、间距与 gap width，不支持完整指骨长度或 shaft geometry。

### 3. Exact edge supervision

对候选 seed `i`，用其 watershed basin 与 GT connected components 的最大重叠得到 `m(i)`：

- 若最大重叠比例低于冻结阈值，`m(i)=unknown`，该 edge 不参与 scorer 监督；
- `m(i) != m(j)`：`y_ij=1`，两候选对应不同 GT 骨骺，应该分开；
- `m(i) == m(j)`：`y_ij=0`，两候选是同一 GT 骨骺的过分裂，不应强制切开；
- 任一端 unknown：ignore。

non-neighbor pair 不再当负例；它不进入 Delaunay edge 集合。bridge corruption 的作用是让同一部署管线在被细桥连接的 soft mask 上仍产生两个 seed，并学习 `y=1`；over-split corruption 产生 `y=0`。scorer 为：

```text
c_ij = P(y_ij=1 | z_dev, sex,
         regional position, spacing, scale ratio,
         seed/basin confidence)
```

骨化中心数量变化通过 `z_dev` 条件下的候选 count/intensity 与 edge confidence 表达；缺少候选时不生成虚拟节点，空图直接 abstain。

### 4. One exact fuzzy-geodesic operator

每条 edge 的两节点局部框按节点间距加固定比例 margin 截取并重采样到 `64 x 64`。seed basin 腐蚀后分别形成固定 source/target seed。

定义 8-neighborhood probabilistic max-pool：

```text
PMaxPool(q)(u) = 1 - product_{v in N8(u) union {u}} (1 - q(v))
```

固定传播：

```text
r_0 = stopgrad(source_seed)
q_t = r_t * p_crop
r_{t+1} = 1 - (1-r_t) * (1-PMaxPool(q_t))
T = 64
Conn_ij = mean(r_T over stopgrad(target_seed))
```

`64x64`、8-neighborhood、crop margin、seed erosion、`T=64` 均只在 original train/val 冻结一次，所有 backbone 共用。无有效 edge 或分母为零时 `E_sep=0` 且不产生梯度。

```text
E_sep = sum(c_ij * u_ij * Conn_ij) /
        (eps + sum(c_ij * u_ij))
```

`u_ij` 汇总 posterior uncertainty、候选稳定性和 scorer calibration。梯度只通过基础模型的 `p_crop`；candidate pipeline 与 prior scorer 全部冻结、stop-gradient。

### 5. Fixed cross-model gradient-ratio interface

在共享输出 logits `l` 上计算：

```text
g_native = ||dL_native/dl||_2
g_prior  = ||dE_sep/dl||_2
alpha = rho * stopgrad(g_native / (g_prior + eps))
L = L_native + alpha * E_sep
```

`rho` 表示 prior 相对 native loss 的目标输出梯度比例；只在 original val 冻结一次，之后所有模型使用同一 `rho`、warm-up 和 confidence rule。若 `g_prior=0` 或无 edge，则 `alpha*E_sep` 定义为 0。主张限于同一输出空间影响比例，不声称不同优化器或 backbone 动力学完全相同。

### 6. Why it handles age, sex and heterogeneous X-rays

- age/sex 形成群体级先验，而不是硬模板；
- image-derived `z_dev` 将群体先验更新为个体成熟 posterior，容纳同龄早熟/晚熟；
- posterior variance 和 candidate instability 触发 abstention；
- variable component count is observed, not fixed；缺少骨化中心不会被强制补出；
- deterministic pose normalization and augmentation consistency prevent exposure/rotation/scale from masquerading as maturity；
- prior only asks whether two prediction-derived candidates should stay separated，不规定完整手形。

### 7. Protocol boundary

先验 encoder、scorer、candidate thresholds、`rho` 和所有算子常量只使用 original train/val。clean-test-v2 在全部冻结后只作一次 R201 final evaluation。本阶段不执行实验。

### 8. Falsifiable scope

若 `z_dev` 相对无图像 developmental conditioning 没有增益，则删除 encoder，论文退化为更简单的 prediction-derived separation graph prior。若候选管线在弱骨化中心上召回不足，则方法应报告 abstention/coverage，而不是增加第二定位网络。
