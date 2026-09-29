# RAM 固定 H0 final-logit 监督对照（2026-09-24）

完整证据：[DIAGNOSIS_AND_ACTION.md](../../outputs/ram_h0_final_target_20260924/DIAGNOSIS_AND_ACTION.md)。

从同一 fresh100 起点、同一固定 384 批次及原 R332 目标出发，仅把新增 H0 正监督由 source logit 改为两步完整 14 成员 final logit。结果：Overlap Dice 从匹配 source 版本的 0.877535 降至 0.877496（−0.003925 个百分点）；Overlap NSD 从 0.787579 升至 0.788246，但 Overlap MSD 略变差。val69 固定 H0 真恢复从 36/71,335 降至 13/71,335；本轮训练病例留出两组都为 0/3,141。final 版本少 1,980 个成员 FP，同时多 1,604 个成员 FN。

首步同批 checkpoint 回放证明最终 H0 目标实际下降、梯度到达现有 decoder/refiner；实现有效但没有目标收益。系数与分母相同，首批新增 final 损失标量比 source 版本小，须作为解释限制。未用 test/TSRS，未调阈值或扩大训练。

目前不值得继续长训这两个局部正监督版本。唯一优先证据工作是固定 H0 与匹配负例的原图—网络输入—成员标注盲态复核；若对齐和数据未见异常，再做冻结特征读出。人工复核尚未进行，不能断言标签错误。
