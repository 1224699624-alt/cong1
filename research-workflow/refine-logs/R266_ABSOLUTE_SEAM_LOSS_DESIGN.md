# R266：非饱和绝对骨缝约束设计

## 动机

R265 的关系项为 `ReLU(margin + p_gap - p_support)`。当两侧骨支持已经很高时，
即使骨缝前景概率仍然偏高，只要满足相对差值，关系项就完全没有梯度。这与本项目的
目标（直接压低骨缝区域的前景概率）不一致。

## 新形式

对每个有效槽位定义：

\[
L_{gap}=\operatorname{softplus}((p_{gap}-\tau_{type})/T_{gap}),\qquad
L_{support}=\operatorname{softplus}((s_{type}-p_{support})/T_{support}).
\]

总关系损失为置信度和关系类型权重加权的
`L_gap + 0.35 L_support`。softplus 保留连续梯度，避免 hinge 在满足 margin 后饱和。

初始上限为：

| 槽位关系 | 前景概率上限 | 权重 | 说明 |
|---|---:|---:|---|
| CC（4 个） | 0.10 | 1.00 | 腕骨间粘连的主约束 |
| CM（1 个） | 0.15 | 0.45 | 腕骨—掌骨基底，降低误切风险 |
| CR（1 个） | 0.28 | 0.15 | 低龄腕骨未完全骨化时保留合法关系 |

年龄/性别没有从验证标签重新读取。它们仍通过 R265 的训练集关系库进入先验图的候选
选择、置信度和槽位激活；R266 仅改变损失数学形式。`clean-test-v2` 未参与。

## 参考思路

- Boundary Loss 用距离变换把边界误差直接纳入优化，与区域损失互补。
- Hausdorff-distance loss 直接惩罚边界偏移。
- 近年的 boundary-wise / boundary-guidance 工作继续采用“区域项 + 边界项”的组合，而不是
  仅依赖 Dice。

本轮先隔离绝对骨缝项；如果原始验证 pilot 显示骨缝概率下降但边界仍模糊，再增加受置信度
门控的 valley-distance 项，避免图像谷值单独决定切割。

## 验证门槛

1. 运行 sanity trainer，确认 `relation_loss > 0`、梯度有限且 CC/CM/CR 均有统计。
2. 小规模 original-val pilot，检查 `gap_probability` 持续下降、`support_probability` 不塌陷。
3. 只有 pilot 通过，才启动完整训练；仍禁止用 clean-test-v2 调参或选模型。

## v2 数值校正

首轮完整训练中发现继承 R265 的 `gradient_ratio=0.04` 会使绝对项权重仅约
`1e-4`，不足以改变骨缝预测。因此 v2 将该比例设为 `1.0`，使加权关系梯度与
基础分割梯度同量级。首轮运行保留在 `r266_absolute_seam`，校正后运行隔离在
`r266_absolute_seam_v2`。
