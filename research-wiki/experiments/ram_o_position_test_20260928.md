# RAM 冻结 O 先验接入位置训练与 test124（2026-09-28）

状态：完整 train425 三组各 1,275 次更新、val69、test124 完成。实际数据源 `G:/gutou/RAM-W600`；测试镜像 248 个文件 SHA 一致。train/val/test 患者数 236/33/66，互不交叉。

共同 C0：R332 fresh100，SHA `ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384`。共同 frozen image-only O generator SHA `e7b738e0b413f9138660475062e76e5d67d9df5c94afc0a2c2597ff0cf126ae5`。比较 `ZERO`、`INPUT_O`（nnU-Net 第一卷积处）和 `REFINER_O`（R332 refiner 第一卷积处），等新增参数量、同 optimizer、同损失、同 batch 增强 SHA，固定末步 checkpoint。零初始化时三步 logits 与 C0 完全一致；两个 O 支路均产生梯度和实际更新。

test124 官方 step2 Overlap Dice：`ZERO 0.874588`、`INPUT_O 0.873789`、`REFINER_O 0.873814`。对应 Overlap NSD@2px：`0.789708`、`0.787901`、`0.788683`。输入位置未减少本轮的重叠漏检或改善标准指标，两个 O 位置均不及同预算零先验。单 seed、三次 train425 遍历与固定 O/接入实现限定结论，不据 test 再选择新位置。

本轮报告：[DIAGNOSIS_AND_RESULT.md](../../outputs/ram_o_position_20260928/DIAGNOSIS_AND_RESULT.md)。原始表：[POSITION_STANDARD_RESULTS.csv](../../outputs/ram_o_position_20260928/POSITION_STANDARD_RESULTS.csv)、[PER_CASE_POSITION_DIFF.csv](../../outputs/ram_o_position_20260928/PER_CASE_POSITION_DIFF.csv)。
