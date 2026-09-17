# 本地先验论文机制综合（2026-07-13）

## 文献集合

来源：`C:\Users\12246\Desktop\先验`，共 6 篇 PDF。

| 文献 | 核心机制 | 对骨骺任务可吸收内容 | 不应直接照搬 |
|---|---|---|---|
| Carneiro-Esteves et al., Neurocomputing 2024, DOI 10.1016/j.neucom.2024.128055 | 用 synthetic connected/disconnected pairs 单独训练 reconnecting regularizer，再以 plug-and-play 方式冻结复用 | 先验独立预训练、合成 topology corruption、跨任务冻结接口 | 原文鼓励前景细线重连；骨骺需要相反的背景骨缝连通 |
| Wyburd et al., Medical Image Analysis 2024, DOI 10.1016/j.media.2024.103222 | TEDS-Net 通过 topology-preserving deformation 变形固定 prior | 将“离散拓扑状态”和“连续几何变化”分开建模；拓扑指标必须独立于 Dice | 固定 prior topology 不适合随发育改变组件数的儿童骨骺；且需按数据调平滑/积分超参 |
| Zhao et al., IEEE JBHI 2024, DOI 10.1109/JBHI.2024.3372576 | SSM 候选与 CNN 候选用 difference score、聚类和 adaptive nearest-neighbor fusion 选择性融合；SSM 中以方差抑制不稳定 landmark | 对高方差关系降低先验权重；先验只能在自身可靠且模型证据不足时激活 | 单一平均形状和 dense mask fusion 对发育异步、可变组件数不适用 |
| Wu et al., IEEE TMI 2024, DOI 10.1109/TMI.2024.3405982 | Masked AutoEncoder 学习 tubular shape prior；Simple Components Erosion 保留拓扑显著修改 | masked corruption learning、外部数据学习后冻结迁移、只处理 topology-critical region | Deep Closing 的目标是前景连接，方向与骨骺相反；simple-point erosion 会阻止我们需要的“把错误合并组件拆开” |
| You et al., IEEE TMI 2024/2025, DOI 10.1109/TMI.2024.3469214 | SPM 将 global/local learnable shape priors 自更新后以 cross-update 注入 CNN/Transformer | 同时建模全局发育和局部异步发育；全局与局部先验必须有不同职责 | dense local shape maps、cross-attention 注入会变成第二分割器并破坏跨模型统一接口 |
| Li et al., IEEE TMI 2025, DOI 10.1109/TMI.2025.3589495 | 快速局部 Euler χ map，通过 χ error 的梯度形成 topology violation map，只在关键区域修正 | 将 prior energy 的梯度显式解释为局部 violation map；只作用于关键骨缝 | 固定 Euler 目标不能描述随年龄变化的正常组件数；额外 correction network 会增加模型依赖 |

## 综合结论

六篇论文共同支持三个方向：先验应独立学习后冻结；先验应只在结构性错误区域介入；跨模型接口应位于输出空间而不是各 backbone 内部。它们同时暴露出儿童骨骺任务不能采用的默认假设：固定平均形状、固定 topology、前景连通以及 dense shape-map 注入。

最值得加入上一版 PD-DSP 的不是另一个形状生成器，而是四个精确升级：

1. 从单一全局成熟度升级为“全局成熟度 + 固定粗区域成熟残差”，表达个体内部 heterochrony；
2. 从单一关系概率升级为带方差/混合分布的统计关系先验，对高变异关系自动降权；
3. 用 masked graph corruption 预训练先验，统一覆盖 bridge、over-split、dropout 和 geometry jitter；
4. 将 max-min separation energy 对 logits 的梯度显式保存为 violation map，只在拓扑关键骨缝施加约束，不增加 correction network。

