# 仓库审计 · 骨缝—骨重叠双先验

审计日期：2026-09-19。审计快照：`737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353`（main，提交记录为 2026-09-17）。

**证据层级：**[仓库事实] 表示实际读取代码/记录；[本轮实测] 仅指随附 CPU 合成探针；[建议] 为本轮新实施要求；[待核] 必须由本地 Codex 查证。文件编号见 `SOURCE_INDEX.json`，不把文档宣称当作真实实验复跑。

## 1. 总结

用户的主线与 R351 experiment wiki 一致：两个数据集、输入骨缝先验、输出骨重叠先验、统一协调、当前只用 nnU-Net；区域指标守住基线，结构指标优先改善。[S02]

工程不是从零开始：有原生无先验 checkpoint 的明确记录、先验模块、局部损失、梯度路由和双数据集 pilot。真正未完成的是**同一原生起点的干净消融、两种先验的信息隔离、TSRS 重叠监督/证据等级、以及部分协调与评价边界**。此前把方法改写为一般 OOF 漏分补全是不恰当的。[S02–S06,S08–S12]

## 2. 原生基线确有记录，但不能宣称本轮验证了权重文件

| 数据集 | 仓库记录的无先验资产 | 记录的 SHA256 |
|---|---|---|
| TSRS | `r317_image_centers_only/r202_two_channel_zero_prior.pth` | `55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b` |
| RAM | `r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth` | `88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9` |

以上路径是 tracker 的尾路径线索，不是本轮确认的绝对路径。[S04] `.gitignore` 排除了 outputs 下的 pth/pt/npy/归档以及原始数据；本轮没有下载/加载这些权重。[S17] Codex 应先在已授权环境定位并校验，而不是重新训练无先验基线。

TSRS 文件名明确带 `two_channel_zero_prior`；必须核查第二通道、转换来源及和原始 R202 的数值等价性。不能因位于 r317 目录就认定它是带先验模型，也不能只凭名字断言 prior-free。[S04,S09]

RAM `NnUNetMultiLabelPrior` 是使用 PlainConvUNet 的仓库多标签实现：原始网络输出 15 通道，前 14 通道用于成员，最后一通道为 prior head，`use_prior=False` 不应用 prior residual。[S19] R325 `plain_native` 的 prior weight 为 0。[S18] 因此可优先作为已有**无先验 nnU-Net 多标签基线**；“完全未经修改的官方 nnU-Net 训练流程”尚不能仅由这些代码推出。无需为修正名称重训基线。

## 3. 四个先验资产的现状

| 先验角色 | 现有依据 | 本轮判断 |
|---|---|---|
| TSRS 骨缝 | R317 image-derived seam；provenance 记录 manifest hash | 优先保留，生成链与覆盖率待本地核验 |
| RAM 骨缝 | R323 RAM-native seam；provenance 记录 manifest hash | 优先保留，注意右腕方向和 padding |
| RAM 骨重叠 | `ImageOverlapPrior`，RAM 多成员 GT 的 `sum>=2` 监督；6 轮记录 | 有真实监督来源和训练记录，可复用/小范围适配 |
| TSRS 骨重叠 | R351 冻结 RAM overlap prior 后迁移；TSRS GT overlap 为 unknown | 有工程弱先验入口，但不能说 TSRS 原生真实重叠监督已经完成 |

来源：[S03,S06,S08,S10,S13]。独立的数据集分支不自动等于独立来源的真实监督。迁移不是天然错误；隐瞒来源、把 unknown 写成负标签，或把漏分当作真实重叠，才会破坏论文表述。

## 4. 必须修正的实现问题

### F01 · 输出侧的 overlap 信息不是独立通路

[仓库事实] `OutputCompletionPrior.refine` 只把 `normalize(seam)*(1-overlap)` 作为第七个特征；没有独立的 overlap 特征通道。当 seam=0，改变 overlap 不会改变该特征。[S08]

[推导] 因此直接把 seam 清零来做“Overlap-only”，会连同输出重叠信息一起消掉。当前 `ImageOverlapPrior(image,seam)` 本身也读取 seam，`SeamInputAdapter(image,seam,overlap)` 又读取 overlap。不能仅切一处开关就声称两个信息源被隔离。

[建议] 输入适配器核心只读取图像和 seam；输出适配器显式读取 overlap，不能只通过 seam 的乘积接收它。干净 C 组需要 image-only overlap generator，或明确标成 legacy 输出位置消融而非无 seam 信息消融。采用 image-only 变体时，需有独立版本、来源/初始化记录、RAM train/val 适配和校验；禁止训练时用 seam、推理时突然置零后冒充同一协议。旧路径保留复现。

### F02 · 现有训练入口不是 native-start 的完整因果对照

[仓库事实] TSRS trainer 固定 `Dataset204...MaturePrior2D`，冻结 backbone，读取第二通道 seam，加载 RAM overlap/refiner；review 与 tracker 确认初始化为 R350，RAM 为 R332。[S04,S05,S09]

[建议] 新入口复用 R202/R325 原生无先验权重；所有关闭先验的输出必须等价于各自原生推理。成熟 R350/R332 作为单独参考保留。A 的 matched continuation 不等于“从头重训 native”。

### F03 · 统一损失的名义与真实调用需对齐

[仓库事实] R350 提供 state mass weighting，但已读 R351 TSRS train_step 使用固定 `.035/.035/.10` 加 ramp；R351 `local_terms` 的 base 还含额外 `.05*foreground_recall`。[S09,S12] 不能说调用了统一质量/占比加权，除非新代码真正调用并记录。

[建议] 所有新 A/B/C/D 保持同一数据集的 base loss、deep supervision 和继续训练协议；结构损失、保真与协调的增量单独登记。统一的是规则，非强制两个数据集的输出通道和可观测标签相同。不得按数据集名字关闭“次要问题”。

### F04 · R349 部分未连接梯度的投影缺陷已复现

[仓库事实] 对辅助梯度 `None` 的坐标，代码直接保留 `None`，即使该坐标的 base 梯度非零。全空间投影需要把该辅助分量视作 0，并在投影后补上必要的 base 方向分量。[S11]

[本轮实测] 原文件副本的 Git blob SHA 与 S11 完全一致。设 `L_base=x+y`、`L_aux=-x`，base 梯度 `(1,1)`、辅助梯度 `(-1,None)`。旧代码投影后辅助与 base 点积约 `-0.5`，`cosine_after≈-0.7071`，仍然冲突。数学上的正确全空间投影是 `(-0.5,0.5)`，点积为 0。详见 `evidence/gradient_none_probe.json`。

这证明该边界实现不满足其投影目标；**不证明它就是历史真实训练退化的原因**。需实际日志统计 None 模式和冲突。还应注意现有 `.15` 是每项 auxiliary 的 cap，不是三项合计 `.15`。[S11] 新代码须明示 individual/aggregate 两个 cap，并记录合成辅助方向。

### F05 · TSRS 局部监督不能用 FN/crowding 替换 overlap 语义

[仓库事实] TSRS 路径 `membership_observed=False`，overlap/positive target 置零；docstring 指明这是 unknown，不是“没有重叠”。forward 仍有 RAM 迁移先验。[S08–S10]

[建议] 将可观测性扩展为显式的空间/成员 mask。TSRS 可在冻结的重叠弱证据范围，用真实二值 GT 做**重叠条件下的投影/前景保真监督**；必须保留 weak 标签、来源和可信度，不能改称真实 multi-membership loss。没有原生成员标注时，真实成员 overlap loss 仍应为 unavailable，而不是造 GT。OOF-FN/crowding 不再是必需阶段。

### F06 · 局部两状态可同时出现在一张图；监督互斥不代表图像问题互斥

[建议] 骨缝支持采用可信 union background 中的 seam 区域，重叠支持采用可观测/弱证据支持的 foreground 区域。这两套监督支持在像素上可定义为互斥，但同一张图、同一窗口可以同时有两者。原始 S/O 概率可以相互冲突，不能强制原始图互斥来掩盖先验错误。

这个简化也只覆盖“union background 骨缝”，不能声称囊括任意骨对在其他骨投影下的全部几何分离关系。

## 5. 指标审计：维持协议，说明含义

### TSRS

R201 `gap_region_fp_rate` 的 ROI 为 `dilate(GT,9×9) & ~GT`，是骨前景周围的背景带，包含外侧边缘，不等于严格的两骨间骨缝。`component_merge_rate` 的单例值为 `pred_components < gt_components`；减少分量可能来自粘连，也可能来自整块漏分。[S14]

因此保留这两个已有结构指标，但正文应说明它们是骨缝/拓扑代理，连同 Dice/IoU、Recall、Boundary F1、逐例可视化和分量计数变化解释。不能依靠“少预测骨头”使结构指标表面改善。若已有可靠匿名实例或真实 seam 标注，可加冻结的 GT-only 配对骨缝验证作为补充；不得运行后重画 ROI 追分。

`evaluate_r201_single_split.py` 会跳过缺失预测，并在尺寸不符时 resize。[S15] 新 runner 先强制核验 case ID 精确一致、尺寸/坐标匹配；不要靠旧 evaluator 容错隐藏缺失或错位。复用其纯指标函数，不启动 `run_r201_unified_eval.py` 默认 clean-test 主程序。

### RAM

R329 协议分别计算：14 成员整体指标、预测/GT `count>=2` 的 overlap map、train 中选出的 top-15 骨对交集。MSD undefined 值不进入均值，同时报告失败率；pair 只纳入 GT 非空的已选 pair/case。[S16]

所以 MSD 降低必须伴随失败率不升、overlap DSC/NSD 不明显变差。pair 指标不单独覆盖 GT 空的骨对假重叠；还要查看全 overlap map 的 DSC/IoU 等。仓库称“official”，本轮核对的是仓库 R329 实现，没有另行验证 RAM 官方发布协议的逐项一致性。

**当前验收重点：**区域 Dice/IoU 是防退化约束；TSRS gapFP/merge、RAM overlap/pair MSD 是结构收益重点，不再把所有结构指标降为无关紧要的附录项。保留全部旧指标，不能为了主张删除不利指标。

## 6. 数据与推理协议风险

- tracker 记录 TSRS 远程 train875 / original-val96；本地 train_labels874 缺 `1833.png`。必须逐 ID 对账，不能静默丢样本。[S04]
- RAM 原生图像仅右/底 padding；右腕 `_R` 已 canonical flip。R323 seam 已在 canonical 方向，不再翻第二次。[S13,S18]
- RAM 首轮禁用增强是现有受控做法。以后重启几何增强时 image/target/S/O/valid/observation masks 需使用同一变换；本轮先不增复杂度。[S13]
- prior coverage、patch/全图 normalization、ignore/padding、TSRS nnU-Net 滑窗/TTA/回采样和标签读取必须锁定。seam 可沿用已验证归一化；overlap 概率不能擅自做每图 min-max。
- RAM overlap prior 在 192×192 上使用面积覆盖软目标训练；上采样图不是像素级真实重叠 GT，也不能称为天然精确校准。[S10]

## 7. 历史结果怎样使用

R351 wiki 报告：TSRS pilot Dice 0.903888861、IoU 0.828067391、gapFP 0.192697431、merge 0.572916667；相对 matched native 的 gapFP/merge 下降约 10.31%/9.84%，但相对初始化 R350 回退。RAM pilot best epoch1 报告 DSC 0.979199379、IoU 0.959733948、overlap MSD 1.829829581、pair MSD 1.654147634；相对 native gate 通过，对 R332 并非全指标提升。[S02]

这些是**仓库已有验证集记录**，不是本轮效果验证；不代表新 native-start A/B/C/D 已完成。正增量 seam guard 是 post-hoc validation 诊断，不自动作为新模型推理策略。[S03,S05]

## 8. 建议执行结论

继续现有工程，不推倒重来；当前只用 nnU-Net。先校验原生资产并修正接口/梯度/局部监督，再完成两数据集 native-start 的 A/B/C/D 和 D-no-routing。TSRS weak-overlap 的真实性与独立原生监督缺口如实报告，不通过改名“补全”绕过去。接口可替换仅作为设计约束，跨骨架效果尚未在本轮验证。
