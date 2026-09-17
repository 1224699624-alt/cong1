# R269 腕骨内部 merge-risk ROI pilot

日期：2026-07-20

## 目的

验证“局部窄颈 + 距离变换鞍点 + 局部盆地/低响应 + X 光灰度谷值”能否在
R110 的单个连通块内部定位腕骨相邻结构的粘连风险。候选生成不读取 GT；
original-val GT 只在候选生成后用于覆盖/误切审计。没有写预测 mask，也没有
使用 clean-test-v2。

## 实现

脚本：`scripts/run_r269_wrist_internal_merge_roi_pilot.py`

- 仅处理 `TSRS_RSNA-Epiphysis/val`；纵向腕骨带为 `v=0.68..0.93`；
- 在腕骨带内的 anchor 连通块中寻找距离峰；
- 用距离图和局部暗响应构造 watershed relief；
- 对 basin pair 计算 neck、saddle、灰度谷、双侧支持四项联合分数；
- 输出紧致 seam ROI 和 12 张可视化；
- GT 审计同时报告真实背景 gap 与实例 ID 接触边界，避免把“两个实例相接”
  和“误切真实骨前景”混为一类。

## original-val 结果

| 项目 | 结果 |
|---|---:|
| images | 96 |
| candidate rows | 161 |
| images with candidates | 56 |
| mean candidates/image | 1.6771 |
| mean union GT background-gap coverage | 0.0411 |
| top-1 mean GT foreground fraction | 0.9588 |
| top-1 mean GT background-gap fraction | 0.0223 |
| top-1 mean inter-instance-boundary fraction | 0.0097 |
| top-1 mean destructive foreground fraction | 0.9525 |

结果说明：联合几何/灰度证据在当前二值 anchor 上仍然主要把同一骨块内部的
形状凹陷当作“两个 basin”，没有可靠定位真实相邻骨缝。R269 暂不具备训练
或 mask-level 推进条件，因此没有启动 R267/R269 长训，避免把明显错误的
ROI 当作先验监督。

## 下一步

不要继续调这套无概率图的硬启发式。下一版应采用“语义概率图 -> 候选
foreground-foreground boundary -> boundary signature classifier”的两阶段
方法：先用成熟 nnU-Net 产生概率图，再让轻量边界分类器判断候选边界是真骨缝
还是同一骨块内部伪边界；训练时使用 train-only / OOF 概率图，验证仍严格按
original-val。候选边界不确定时 abstain，不强行切割。
