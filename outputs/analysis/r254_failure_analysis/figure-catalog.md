# R254 图目录

## Figure 1 — directional metric change

- 文件：figures/figure-01-directional-metric-change.pdf
- 目的：区分真正的结构改善与表面上好看的指标。
- 数据：96 张 original-val 的方法均值相对 native 均值；已把方向统一成正值代表改善。
- 观察：HD95、ASSD、component MAE 为正，但 gap FP、Boundary IoU、Surface Dice、Precision 为负。
- 含义：减少极端碎片/远距离错误不等于解决近距离骨缝。
- 限制：单 seed，未画 seed 误差条。

## Figure 2 — paired gap FP and Dice

- 文件：figures/figure-02-paired-gapfp-dice.pdf
- 目的：显示同一图像上的变化方向。
- 数据：96 对图像；虚线为两方法相等。
- 观察：gap FP 大量点位于恶化侧，而 Dice 变化较小且混合。
- 含义：总体 Dice 掩盖了骨缝专门失败。
- 限制：original-val 用于模型选择。

## Figure 3 — training confidence dynamics

- 文件：figures/figure-03-training-confidence-dynamics.pdf
- 目的：检查损失是否按预期改变训练锚点置信。
- 数据：15 个 HA-PDSP+PCR epoch。
- 观察：selected negative probability 快速下降，bone-core probability 上升并保持。
- 含义：优化器执行了指定目标，但目标区域与真实近缝失败错位；问题在 anchor 定义而不在 loss 是否生效。

