---
type: experiment
node_id: exp:ram_o_source_head_20260924
added: 2026-09-24
---

# RAM O 先验在困难位置及单一 source-head 接入验证

完整证据：[DIAGNOSIS_AND_ACTION.md](../../outputs/ram_o_source_head_20260924/DIAGNOSIS_AND_ACTION.md)。仅 RAM train425 / val69；test 和 TSRS 未运行。

冻结的 image-only O 在 val69 固定 H0 困难成员位置均值为 0.2454，只有 15.07% 的 H0 唯一像素达到 0.5。与原图局部强度/梯度及 C0 预测摘要共同建模时，O 的条件诊断 AUC 增量为 +0.00585，低于预锁定的 0.01 实用筛选线。原 C0 的 O refiner 输入首层权重全零，因此没有 O 信息流。

探索性单点对照把 O × 零初始化成员条件 adapter 加到原生 source head。ZERO 与 REAL 从同一个 fresh100 和 optimizer 状态开始，匹配 384 个 batch，ZERO 与旧 SRC_F0S1 逐参数及标准指标完全一致。REAL 的 adapter 有非零梯度和更新，但 val H0 最终正确成员恢复仍为 36/71,335（ZERO 也是 36），Overlap Dice 0.8775353 → 0.8775333，Pair Dice 0.8354676 → 0.8354576，NSD 也未改善；有效区域 FP 多 40 个。内部响应未转化成标准收益。

结论是这一个 O 表示/接入/384 步组合没有价值，不能晋升为成熟版本。下一步仅优先只读复核低 O 的 RAM H0 原图、GT、生成器输入和热力图，确认先验覆盖不足的可复现原因后再考虑一个单因素修正；不继续盲换网络层。
