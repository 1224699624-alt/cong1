---
title: "TSRS 骨缝分离的拓扑、边界与教师保真文献审查"
type: literature-review
date: 2026-08-18
status: working
---

# 结论

TSRS 的下一步不应继续增加 bone-support loss。R279--R283 已经显示 seam suppression 与 support restoration 存在结构性冲突。文献更支持三个彼此分工的机制：

1. 在局部骨缝 corridor 内，对**背景/骨缝通道**施加拓扑连通约束；
2. 用 signed-distance/boundary loss 直接控制边界漂移、HD95 与 ASSD；
3. 只在非干预、高置信区域使用冻结基线的一致性保真，不能用教师前景概率单独推断真实重叠。

# 代表性论文

| 论文 | 发表信息 | 对 TSRS 的启示 |
|---|---|---|
| Shit et al., *clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation* | MICCAI 2021; arXiv: [2003.07311](https://arxiv.org/abs/2003.07311) | clDice 用软骨架 precision/recall 约束连通性。TSRS 不能直接对前景骨头使用；应在局部 seam corridor 的背景补集上反向使用。 |
| Kervadec et al., *Boundary loss for highly unbalanced segmentation* | MIDL 2019; arXiv: [1812.07032](https://arxiv.org/abs/1812.07032) | 距离变换边界项直接针对 interface，而不是继续增加 support。适合小系数、局部 corridor 使用。 |
| Karimi & Salcudean, *Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks* | IEEE TMI 2020; arXiv: [1904.10030](https://arxiv.org/abs/1904.10030) | HD 型项能针对远端边界离群点，但应小权重、先做 smoke，避免与 seam loss 同时过强。 |
| Tarvainen & Valpola, *Mean teachers are better role models* | NeurIPS 2017; arXiv: [1703.01780](https://arxiv.org/abs/1703.01780) | teacher consistency 应是置信度/区域选择的保真约束，不应把 teacher 的单一类别判断当作结构状态标签。 |
| Xia et al., *Uncertainty-aware multi-view co-training for semi-supervised medical image segmentation and domain adaptation* | MICCAI 2020; arXiv: [2006.16806](https://arxiv.org/abs/2006.16806) | 不确定区域应降权或 abstain；支持 uncertainty-masked consistency，而非全图蒸馏。 |
| Kirchhoff et al., *Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures* | 2024 preprint; arXiv: [2404.03010](https://arxiv.org/abs/2404.03010) | 骨架召回比全图 support 更接近“不能断开 seam channel”的目标，可作为 clDice 的高效实现参考。 |
| Zhou et al., *GLCP: Global-to-Local Connectivity Preservation for Tubular Structure Segmentation* | 2025 preprint; arXiv: [2507.21328](https://arxiv.org/abs/2507.21328) | 全局拓扑与局部断裂应分开建模；对本项目的启示是 local seam-discontinuity，而不是 generic two-sided support。 |

# 与既有实验的合并判断

- R279：单向 seam suppression 改善 merge，但造成前景收缩。
- R280/R281：support 恢复召回，却填回 seam；手工降权仍未解决。
- R282：GradNorm 解决量纲，未解决 support 的空间目标错误。
- R283：空间/置信度 support gate 避免 seam-buffer 泄漏，但仍损伤 Dice/Recall/HD95。
- R344/R345B：区域 gate 或 teacher-foreground conflict gate 会把“尚未切开的真实粘连”和“危险误切”混为一类。

因此 support loss、teacher-foreground conflict gate 和全局 scalar gate 均不应作为下一轮主机制。

# 预定下一方案（未启动）

\[
L = L_{seg} + \lambda_t L_{bg\text{-}topo}^{corridor}
  + \lambda_b L_{boundary}^{corridor}
  + \lambda_p L_{preserve}^{outside}
\]

- `L_bg-topo`: 对 GT-derived training seam corridor 的背景补集使用 soft-clDice/skeleton-recall；推理只使用图像和冻结 seam prior 生成 corridor，不读取 GT。
- `L_boundary`: corridor 内 signed-distance boundary loss；不加入 bone-support restoration。
- `L_preserve`: 仅在 corridor 外、teacher 高置信且不确定性低的区域做一致性；不使用 teacher foreground 作为 overlap 标签。
- 不把 TSRS 的颜色 ID 当成真实 overlap prior；RAM 的实例重叠仍是独立分支。

先进行 train-only/2--4 image overfit smoke，锁定拓扑项和边界项的梯度占比，再做一次 18 epoch original-val R201 受控实验。若 Dice/IoU、Boundary、Surface、HD95/ASSD 与 interaction hard gates 不能同时通过，则关闭这条 loss 微调路线，不再重复 support/gate 变体。
