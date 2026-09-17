# R271 indexed-instance label correction

日期：2026-07-20

## 发现

`data/raw/TSRS_RSNA-Epiphysis/train_labels` 和 `val_labels` 的 PNG 是 `P`
模式 indexed-palette 标签，不是二值 mask。像素值是每张图内部的实例 ID，
颜色来自 PNG palette；例如样例图存在 30 个索引值，其中 0 是背景。不同图像
之间 ID 不能直接当作固定解剖类别，但同一图像内可以可靠地区分相邻实例。

此前 R269 的 anchor 仍然是二值预测 mask，因此没有利用 GT 的实例 ID；这也是
候选边界召回很低的重要原因。

## 新的隔离目标

新增脚本：`scripts/prepare_r271_instance_boundary_targets.py`

输出目录：`outputs/targets/r271_instance_seam_full`

生成内容：

- `instance_id/`：原始实例 ID 的无损 numpy 备份；
- `semantic/`：所有骨的前景目标；
- `boundary/`：两个正实例直接接触处的边界带；
- `seam/`：背景中最近实例 Voronoi 归属发生变化的骨缝中心线；
- `seam_probability/`：以 seam 中心线为中心的软骨缝热图；
- `manifest.json`：样本数和实例统计。

Voronoi seam 采用背景距离变换和最近实例 ID 场，不使用固定实例编号做分类，
因此可适应每张图中骨化中心数量不同的情况。

## 规模

- train：875 张，平均 25.97 个实例/图；
- original-val：96 张，平均 25.56 个实例/图；
- `max_gap_radius=12`，`seam_sigma=2`；
- 原始数据集未修改；clean-test-v2 未使用。

## 解释

这些 seam_probability 是由 GT 实例 ID 构造的训练监督/审计目标，不是测试时可
直接使用的 GT 概率图。训练时应让 nnU-Net 或边界分支学习预测该热图；推理时只能
由 X-ray 和模型 softmax 产生同类概率。下一步应使用该实例边界监督训练 softmax
与 boundary head，再把预测 seam prior 接入 separation loss。
