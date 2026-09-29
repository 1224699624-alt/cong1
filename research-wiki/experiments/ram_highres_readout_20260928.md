# RAM 冻结高分辨率特征读出（2026-09-28）

状态：P0/P1 完成；新的完整分割修正 `NOT_RUN`。

实际数据源 `G:/gutou/RAM-W600`，train425/val69 与服务器 988 个对应文件 SHA 一致。按 `Metadata.xlsx` 的 PatientID 建立 train 内 fit256/cal85/holdout84 病人隔离分组，val69 独立。C0 为 R332 fresh100，SHA `ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384`，全程冻结。

P0 使用成熟 R332 公共信息及 `refiner.dec1` 全分辨率隐藏特征；P1 在同容量读出器中将额外投影块改为 nnU-Net source 头前的 `decoder.stages[4]` 特征。两者各 1300 次相同步骤。train 内 calibration 按完整自然 Q 的负例 FPR≤0.001 定阈，再固定用于 holdout/val。

val69 自然 Q 的 P0/P1 AP（16,384 桶近似）为 0.123940/0.119111；精确 H0 选中 26,981/26,975（完整 H0=71,335），精确 FP 165,848/166,974。P1 没有稳定增量。读出分数未写回分割，Dice/NSD 新成绩 `NOT_RUN`；不能据此断言所有高分辨率层无信息。

唯一下一步：固定 train 病例坐标，对 H0 与同病例同成员高分 Q 假阳性作盲态原图、实际输入、原标注复核，状态 `NOT_REVIEWED`。后续修正对照未获本轮证据支持，`NOT_AUTHORIZED / NOT_RUN`。

完整报告与原始证据：[DIAGNOSIS_AND_NEXT_ACTION.md](../../outputs/ram_highres_readout_20260928/DIAGNOSIS_AND_NEXT_ACTION.md)、[PROBE_RESULTS.csv](../../outputs/ram_highres_readout_20260928/PROBE_RESULTS.csv)、[PROBE_TRAINING_TRACE.jsonl](../../outputs/ram_highres_readout_20260928/PROBE_TRAINING_TRACE.jsonl)。
