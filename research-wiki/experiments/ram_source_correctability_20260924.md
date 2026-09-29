# RAM 困难漏分可修正性（2026-09-24）

完整证据见 [DIAGNOSIS_AND_ACTION.md](../../outputs/ram_source_correctability_20260924/DIAGNOSIS_AND_ACTION.md)。本轮仅 RAM train425 / val69；无 test、TSRS 或新先验分支。

- 起点为成熟 R332 组合 fresh100，SHA256 ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384，配套 AdamW/scaler/step100 已恢复。
- 零更新 H0：val69 有 71,335 个成员漏分；理想只修这些成员，原评价器 Overlap Dice 0.877243→0.906992（+2.9749 个百分点）。这是 oracle 诊断，不是模型成绩。
- 匹配四组各 384 更新：F0S0 0.877187、F0S1 0.877535、F1S0 0.876616、F1S1 0.877128 Overlap Dice。F0S1 相对 F0S0 +0.0348 个百分点，但 Overlap NSD 下降；F1 开放 decoder D2 没有净收益。
- 固定 H0 的 source 移动未转成目标恢复：F0S1 val69 终点仅恢复 36/71,335，训练病例留出 0/3,141；相对 F0S0，全图成员 FN 少 5,546、FP 多 5,967。第 1/192/384 步只读回放、真实 Q_H 首步前后差及同批匹配真负 source BCE 均已保存。
- 不建议直接扩大到完整 train425 长训。唯一优先小对照是在其余条件不变时，将固定 H0 新增监督由 source logit 改为两步 final logit。未完成人工标注复核，不能归因为图像或标签错误。

运行锁、真实轨迹、逐例标准结果、代码 diff 与预算记录均在 outputs/ram_source_correctability_20260924/。本轮不延续旧任务队列。
