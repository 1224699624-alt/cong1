# Final Prior 短 pilot 冻结协议

2026-09-19，用户已授权登录服务器并推进 native 复现与短 pilot；未授权自动进入长训练。本协议在 pilot 结果产生前记录。

## 数据与资源

- 服务器：connect.westc.seetacloud.com:52807，工作区 `/root/autodl-tmp/YOLO_SAM_generic_src`。
- TSRS 缺失样本 1833 已在服务器找到，原图与标签均存在；沿用完整 875 train / 96 val，不启动 874 备选方案。
- TSRS `splits_final.json` SHA256：`21f9272b992bdd17e2ae13aa7bd977f6dada2461c9c0e04e33cbabf1648f29f4`。
- RAM：425 train / 69 val，R323 prior 覆盖完整。
- 原始数据与历史结果不改动。新输出在 `/root/final_prior_v1`，初始系统盘可用约29GB，数据盘约4.7GB；目前无需删除文件。
- 数据审计只读 train/val，没有读取 test。核对 case ID 及图像文件内容的 train/val 交集；未将此宣称为独立患者身份审计。

## 同源控制

- TSRS native：R317目录下 `r202_two_channel_zero_prior.pth`，SHA `55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b`，current_epoch=228，第二输入通道权重全零。
- RAM native：R325目录下 `plain_native_best.pth`，SHA `88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9`，row.epoch=23。
- A/B/C/D backbone 均严格加载对应 native；R332仅允许作为输出prior的identity初始化来源，不能作为backbone来源。
- TSRS backbone 的第二输入通道始终置零，否则 joint native_continue 可能自行学会利用 seam，污染 A 控制组。seam 只进入插件。
- 所有变体均构造相同模块后按开关启用，避免随机初始化顺序改变 backbone 起点。

## 运行预算与顺序

1. 完整 native validation 复现。
2. 单元测试；adapter_only 实际 batch identity/gradient smoke。
3. joint_finetune 实际 batch smoke，通过后进行 matched pilot。
4. TSRS A/B/D；RAM A/C/D；每组 2 epoch × 100 iteration，单 seed。

adapter_only + A 没有可训练参数，不能冒充 native_continue。因此 adapter_only 仅作插件实现检查；有实际继续训练的匹配短 pilot 使用 joint_finetune，不把这轮称为 adapter-only 效果消融。

所有同数据集arms固定 AdamW、backbone LR=2e-6、prior LR=2e-5、weight_decay=1e-4、同 polynomial schedule（保持LR比例）、同batch大小与数据增强、同seed与迭代数。输出residual scale=1，overlap context scale=1；不根据test调整。TSRS strong overlap loss关闭。具体运行精度及clip参数以审核后的统一run manifest为准。

TSRS沿用nnU-Net EMA foreground Dice选checkpoint；RAM用val overall DSC选checkpoint。最后所有标准指标必须来自该单一checkpoint，记录epoch和SHA。TSRS最终整例推理统一用原nnU-Net predictor及R201 evaluator；RAM统一复用原official_metrics。开发诊断不得替换主指标。

## 解释边界

native复现与历史数值差异如实记录。TSRS本机复现Dice差8.4e-7、IoU差1.46e-6、HD95无差异，不声称逐位相同；本机新native结果作为pilot额外参考，主要因果比较仍为同轮次A对B/D。RAM完整69例已比较的历史指标逐项一致。

短pilot不等于最终结论，不自动运行full/3-seed/test，不发明R352/R353。若梯度、身份或checkpoint绑定检查失败，先修复实现再启动；若pilot出现明显系统退化，报告并停止向长训练推进。
