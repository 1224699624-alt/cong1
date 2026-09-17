# Round 4 Refinement

## Problem Anchor

- Bottom-line problem: 建立一个与基础分割网络解耦的儿童手部解剖先验，使同一先验在 nnU-Net、ARAA、YOLO+SAM、DINOv3/U-Net 等不同模型上都能改善骨缝粘连、边界质量和解剖一致性。
- Must-solve bottleneck: 儿童骨骺随年龄、性别和个体成熟度高度异步变化，固定模板会把正常发育差异误判为分割错误。
- Non-goals: 不设计新 backbone；不预测完整 foreground atlas；不使用 clean-test-v2 调先验；不鼓励骨头前景连通；不假设实例 ID 固定。
- Constraints: 仅 TSRS_RSNA-Epiphysis；R201；原始数据不变；先验和集成规则跨模型冻结；必须先回连并审计 RSNA metadata。
- Success condition: 同一冻结先验在至少 4/5 个异构模型上改善骨缝、边界和组件指标，同时 Dice/IoU/Recall 非劣。

## Anchor and Simplicity Check

- 唯一贡献仍是发育条件化、prediction-derived 的负空间关系能量。
- X 光仅输出全局成熟 posterior；先验不输出位置、support map 或 mask。
- 候选、seed、边均由基础模型 soft mask 经固定 stop-gradient 管线产生。
- 本轮只替换有重复累计问题的传播递推，并补全梯度权重数值保护；其余 Round 3 定义保持不变。

## Final Proposal: Prediction-Derived Developmental Separation Prior (PD-DSP)

### Developmental conditioning

部署先验使用低维 `q(z_dev | GlobalPool(g(x)), sex, chronological_age_if_available)`。RSNA `boneage` 只能作为 original-train 的可选 radiographic-age ordinal teacher，不能冒充 chronological age，部署时删除 teacher head。sex 是弱协变量；pose、尺度、左右手和曝光扰动通过确定性归一化及增强一致性处理。

### Shared candidate graph

clean GT、合成 bridge/over-split corruption、backbone 训练输出和推理输出均以相同 soft-mask 表示进入：固定 probability normalization、distance transform、peak threshold/min-distance、watershed、hand-relative coordinates、Delaunay 和长边删除。整个管线 stop-gradient。

节点默认无具体解剖名称，仅使用区域位置、面积、间距、尺度比和候选置信度。当前标签只支持骨骺中心 presence、面积、相对位置、间距与 gap width，不支持指骨 shaft length。

### Provenance-correct relation supervision

候选 seed/basin 与原始 GT component provenance 做唯一最大重叠匹配。不同 GT ID 的 edge 标为 should-separate，同一 GT ID 标为 over-split negative；背景、混合或无法唯一归属的候选一律 ignore。non-neighbor 不作为负例。

冻结 scorer：

```text
c_ij = P(should separate | z_dev, sex,
         regional geometry, scale ratio, candidate confidence)
```

缺少骨化中心时不生成虚拟节点；空图 abstain。组件数量变化通过 candidate count/intensity 和 developmental posterior uncertainty 表达。

### Exact max-min fuzzy connectivity

每条 Delaunay edge 的局部框按两 seed 间距与局部尺度的固定公式扩展，统一重采样到 `64×64`。source/target seed 来自同一 watershed basin 的固定腐蚀。

```text
r_0 = stopgrad(source_seed)
r_{t+1} = max(
    r_0,
    min(p_crop, MaxPool3x3(r_t))
)
T = 64
Conn_ij = mean(r_T over stopgrad(target_seed))
```

- `MaxPool3x3` 是固定 8-neighborhood hard max；`min/max` 是逐元素标准运算；padding 固定为 0。
- 在 `64×64` crop 上，`T=64` 覆盖最大 Chebyshev 距离。
- 递推计算 strongest max-min foreground path：连接值等于最优路径上的最低前景概率；低背景响应不会因循环或重复迭代累积饱和。
- 运算分段可微，梯度只进入最强连接路径上的 `p_crop`。

```text
E_sep = sum(c_ij * u_ij * Conn_ij) /
        (eps + sum(c_ij * u_ij))
```

`u_ij` 由 posterior uncertainty、候选稳定性和 scorer calibration 决定。无有效 edge 时 `E_sep=0` 并跳过 prior 梯度。

### Fixed cross-model gradient-ratio integration

所有模型在同一最终输出 logits `l` 上计算：

```text
g_n = ||dL_native/dl||_2
g_p = ||dE_sep/dl||_2

if no_valid_edge or g_p < tau_g:
    L = L_native
else:
    alpha = clip(
        rho * stopgrad(g_n / (g_p + eps)),
        alpha_min, alpha_max
    )
    L = L_native + alpha * E_sep
```

batch reduction、`rho`、`eps`、`tau_g`、`alpha_min/max`、warm-up 和 confidence rule 只在 original train/val 冻结一次，随后跨 backbone 共享。`alpha` stop-gradient，不引入二阶梯度。

### Why this prior handles pediatric variability

- 年龄/性别只定义群体分布，不定义硬模板；
- X 光的低维 `z_dev` 区分同龄同性别的早熟、晚熟和异步成熟；
- posterior variance 在不确定病例上降低约束；
- variable component count 由观察到的候选图表达，绝不强制补齐固定 slot；
- 先验只判断 prediction-derived pair 是否应保持背景分隔，不规定完整手形，因此可以冻结后接入不同模型。

### Protocol and falsifiability

全部先验参数、候选规则和梯度比例仅用 original train/val；clean-test-v2 只在协议冻结后作一次 R201 final evaluation。本阶段不运行实验。若 `z_dev` 对无图像条件版本无额外价值，则删除 image encoder；若候选召回不足，则报告 coverage/abstention，而不是增加第二定位网络。
