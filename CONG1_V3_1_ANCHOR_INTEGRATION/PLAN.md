# V3.1 / Round 4 修订：保留成熟单先验能力，再完成双先验整合

**日期：2026-09-20。工作根目录：`G:\gutou\YOLO+SAM`。当前骨架：仅 nnU-Net。**

本文是基于新上传“Current Best Experiment Map”及 Round3 结果对上一份 `CODEX_V3_ROUND4_MECHANISM_DIAGNOSIS.md` 的增量修订，不代表第四轮已经执行。保留 V3 的输入定义、信息隔离、正确梯度实现、数据清单和评价边界；把执行优先级改为“成熟资产保真接入 → 小型受控比较 → 双先验协调”，不从头重写工程。

文中 [S1]–[S6] 均为用户提供的文件，见末尾来源表。**历史结果只按报告转述；本文件没有重新加载本地 checkpoint、读取真实训练代码或运行模型。所有新增组名和检查要求都是拟执行方案。**

## 0. 不得改变的研究对象

TSRS 和 RAM 均保留以下框架：

- 输入侧：原始归一化 X-ray 与空间对齐骨缝热力图联合输入 nnU-Net；B/D 为 `[image, seam]`，A/C 为 `[image, zero]`。
- 输出侧：骨重叠先验约束成员关系及重叠区域分割。不是一般前景补全、OOF 假阴性修补或 crowding 检测。
- 训练侧：基础分割损失、局部骨缝/重叠/保真约束，以及正确实现的梯度协调。
- 一张图可以同时存在骨缝和骨重叠。TSRS 重点看骨缝，RAM 重点看重叠，但不据数据集名称关闭次要先验。
- 目标是整体 Dice/IoU 守住质量底线，同时主要结构指标取得有意义的提升；不以追求大幅 Dice 涨点替代结构目标。
- 可替换骨架是接口目标，本阶段不运行其他骨架。

`[adapted_image, zero]` 不是本轮联合输入定义。seam loss 不能代替输入热力图。输出模块即便历史类名含 completion，也只能按实际重叠成员作用解释，不能据类名把论文改成一般补全。

## 1. 新报告改变了什么

[S1] 同时提供了成熟结果和 clean pilot 的定位。最重要的不是“哪个编号更新”，而是区分三种证据：

1. **成熟效果证据**：R317 的 TSRS 骨缝结构表现、R332 的 RAM 重叠表现、R350 的 RAM 统一优化验证结果。
2. **公平接入证据**：Round3 的 native-start matched A/B/C/D；其中 TSRS 仅 A/B，RAM 完整 A/B/C/D。
3. **未完成的论文结论**：两个数据集各自完整双先验，以及统一协调对该完整框架的独立贡献。

新上传 `PILOT_RESULTS(1).csv` 与上一份 `PILOT_RESULTS.csv` 字节相同；两份 `PAIRED_DELTAS` 也相同。本次主要增加历史设计与效果的定位，不是又完成了一轮涨点训练。见 `ATTACHMENT_CHECK.json`。

[S1] 中“当前论文核心证据仍是两条成熟单先验线”，在这里仅理解为**当前已支持的证据范围**，不能理解为把用户最终双先验论文改成两个互不相干的单先验课题。

## 2. 资产分工：保留能力，而不是拼最佳数字

| 资产 | 新报告支持的身份 | 本轮用途 | 不能据此声称 |
|---|---|---|---|
| TSRS R317 continuous α=.035 | 成熟单骨缝效果参照；旧 baseline 推理链不完全 matched | 保留热力图、联合输入、连续 separation 定义，定位新旧行为差异 | 已证明在当前 matched N0 上的纯先验增益 |
| RAM R332 step2 | 报告中最强的 locked-test 单重叠结果 | 保留逐成员关系、两步迭代及已有 surface/volume 设计；作为输出侧整合基础 | 124 test 与69 val 可直接减分排名；所有子设计已分别被因果证明 |
| RAM R350 epoch29 | 成熟 validation 统一状态/协调组合结果 | 精确复核损失、参数组、初始化和梯度处理，作为后续协调方案参照 | 路由单独贡献已证明；TSRS 同样成功 |
| Round3 V3 | 联合输入、独立 O、信息隔离、修复后的路由和 matched pilot | 保留干净接口和测试，不因效果混合删历史 | 已完整复现 R317/R332 的学习能力；最新即最好 |
| 历史 R351 | mature init、旧耦合、迁移证据的探索 | 只读历史定位，不改表、不覆盖 | clean native-start 双先验已成功 |

R317 checkpoint 线索（按 [S1]，以本地实际文件和哈希核验）：
`outputs/nnunet/r317_image_centers_only/nnUNet_results/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_r201_selected.pth`

R332 checkpoint 线索：
`outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth`

R350 的具体本地 checkpoint 路径未在本次报告中完整给出。沿本地实验记录及 `outputs/analysis/latest_best_vs_baseline_20260824.csv` 定向定位，不猜路径、不按文件修改时间挑“最新”。

### 2.1 历史数值必须连同参照与 split 保存

- R317：Dice .902858，IoU .826369，Gap FP .181737，Merge .5000，HD95 14.0084 px，ASSD 4.0475 px。[S1 §1]
- R332 step2 对 R325，**locked test 124例**：Overall DSC +.000523，Overlap DSC +.003895，Overlap NSD +.009349，Overlap MSD −.1050 px，Pair MSD −.25676 px。Overlap RAVD、pair MSD fail rate 的副作用保留。[S1 §4]
- R350 RAM epoch29 对 R332 step2，**validation**：Overall DSC +.0000206，Overlap DSC +.0007594，Overlap NSD +.0008252，Overlap MSD −.01004 px，Pair MSD −.00878 px。[S1 §6]
- Round3 B/C/D：继续以同预算 A 为主参照，N0 为补充参照；不替换历史 gate 结论。[S2–S5]

历史改善量不能相加成一个“最终 D 的收益”。不同 split、初始化、训练预算、预处理或聚合口径的数字不能直接组成公平主表。

## 3. 先明确设计差异，不先指定退化原因

只针对下表追踪本地相关函数，不再全盘搜索所有实验。

| 维度 | 需要对照的成熟设计 | V3 要核实的内容 |
|---|---|---|
| TSRS 输入 | R317 `[image, continuous seam]` | 已通过的联合输入不重做；核验归一化和缓存版本是否一致 |
| TSRS 局部优化 | R317 continuous separation α=.035 | preserve、state-mass attenuation、投影、per-aux/merged cap 叠加后的有效梯度，不把相同 α 当成相同优化强度 |
| RAM 成员信息 | R332 image、own probability、other-member max/sum、uncertainty、gradient | 当前 refiner 特征是否保留、顺序/尺度是否相同；不能根据摘要遗漏就宣布代码已删除 |
| RAM 迭代 | R332 step2，联合多步训练 | V3 步数、参数共享、每步 feature 重算、梯度 detach 是否一致 |
| RAM 几何目标 | R330–R332 surface/volume 正则 | 实际是否复用、具体公式、作用对象、权重和分母；缺失时标为设计差异，不立即宣布因果 |
| 初始化/可训练性 | 成熟 refiner 及其来源 backbone | V3 fresh zero-residual；backbone 是否共同训练、学习率是否相同 |
| 训练暴露 | 历史有效更新量与采样 | Round3 每组200 updates，不自动等价于完整遍历2个epoch |
| 协调 | R350 状态/参数分组与旧路由 | 只保留有效思想，旧 None 缺陷不得复制；新 merged cap 也不是历史原规则 |

必须区分：

- **先验生成器有信息**，不等于它在 native 错误处有足够有用的信息。
- **C/D 输出方向混合**，不等于已经证明 BCE、fresh init、路由或先验校准中的某一项就是原因。
- **R332 更成熟**，不等于其所有设计必然比 V3 更好；需要保真复用与对照。

新报告对 downstream 的解释作为诊断线索保留，但不要把生成器质量、O 对难例的覆盖或分布差异永久排除。[S1 §5、§Where stuck；S2]

## 4. 两条实验线同时保留，表格不得混用

### 4.1 Native-start 因果线

N0 是已有无先验 nnU-Net。A/B/C/D 从同一个 backbone N0 出发，按同一追加训练预算、采样、增强、优化器、可训练性、选模和 evaluator 比较。

C/D 可使用同一份、来源合法且明示预训练成本的成熟重叠模块；公平性**不等于所有先验都必须从随机权重重新训练**。但不得把完整 R332/R350 的 mature backbone 冒充 N0，也不得让 C/D 使用不同来源 refiner。

仅有相同追加训练预算，不意味着先验预训练的总计算成本相同。总成本、使用过的标签和初始化均需另列；必要的容量控制留到最终主张阶段，不在本轮展开。

### 4.2 Mature-start 能力保留线

从已验证的成熟组合出发，所有对照共享该组合的**完整初始化**和相同追加预算，回答“新增信息/先验/协调是否保住并改善已有能力”。

- RAM 可以从 R332 的实际 backbone + refiner 组合开始。
- TSRS 可以在重叠路径依据清楚之后，从 R317 开始比较。
- 命名必须用 `M_` / `T_` 等前缀，绝不叫 native A。
- 成熟模型保留组也要继续训练同样预算；冻结原 checkpoint 单独记为 `M0` / `T0`。

这条线不是取消公平消融，而是把公平性放在正确的问题上：**相同成熟起点的增量效果**。它不能代替 native A/B/C/D 的归因，也不应因含成熟权重而被全部降为无用结果。

## 5. 阶段 P0：只做必要的成熟资产与验证口径对齐

优先使用现成、绑定 case ID/权重/evaluator 的 validation 结果；缺失时做固定权重推理，**本轮不开 test**。

1. R317：恢复历史原始前向和预处理记录；在现有96例验证清单输出结果。不得使用旧 unmatched baseline 计算新因果增益。
2. R332、R350：读取已有69例 validation 结果，缺失时分别固定推理。不要把 R332 的124例 locked-test 数字直接放进本轮69例表。
3. 核验 raw/preprocessed 坐标、padding、canonical flip、阈值、后处理及距离单位。不要静默 resize 掩码、丢 case 或换 evaluator。
4. 先分别保留“历史函数复现”和“当前统一验证评价”；两者不一致时记录来源，不能调参数让数字凑上。
5. 输出 checkpoint/component/init graph。确定某个 bundle 内哪些是 backbone、refiner、generator、optimizer；不可凭文件名互换。

现有875/96、425/69、1833显式来源已在 Round3 锁定，未变化不重审。RAM runner 的历史运行时 SHA 缺口保留披露；新运行前归档实际代码，不能把事后重构 hash 写成过去运行时记录。

## 6. 阶段 P1：R332 输出能力的函数保真接入

这是本修订新增的最高优先级工程动作。**只改接口，不同时改算法、loss、步数、阈值或训练制度。**

### 6.1 先保住旧函数

将实际 R332 backbone + per-member refiner 组合封装到 V3 接口，保留：

- 14个独立 sigmoid 成员，不改 softmax、不压 union；
- 原有 own/other-member 特征定义和顺序；
- 两步迭代，逐步重算与原有训练/推理语义；
- 空间坐标与原有 residual bound 等已存在规则；
- 成熟 refiner 权重，不能把最后残差层重置为零。

新增 seam 输入通道的零初始化、或新增 O 特征权重的零初始化，与“清零已训练 refiner”是两回事。

在新 wrapper 禁用新增输入作用时，应满足：

`F_R332_legacy(image) ≈ F_R332_port(image, seam=0, O_extension_disabled)`。

比较每步成员 logits、最终 mask、各中间张量的形状和通道定义，并检查已有验证集指标。中间维度按实际代码核验，不根据摘要猜测。数值容差依据相同 dtype/设备的数值复现要求提前写定；它不是放宽区域性能 gate。出现不可解释差异时先修映射，不训练补救。

### 6.2 独立 O 通道如何接入

已经训练好的 image-only generator 保留，不重训：
`e7b738e0b413f9138660475062e76e5d67d9df5c94afc0a2c2597ff0cf126ae5`。

如果旧 refiner 与扩展版除了新增 O 特征外完全兼容，可按实际第一层权重构造：

- 原特征对应权重逐张量复制；
- 仅 O 对应的新权重初始为零；
- 其余参数及两步迭代完全保留。

若结构不兼容，不允许 `strict=False` 大量跳键后宣称继承 R332；先交付不启用 O 扩展的函数保真版本。当前只做最小兼容修改，不为凑同一个类名删掉成熟设计。

**R332 已训练的成员重叠关系模块与显式 O 概率图不是同一对象。**本阶段保留前者，再验证后者是否提供增量信息。新增 O 支路在零初始化时不影响结果是预期行为，但训练后若仍无有效作用，不得宣称显式 O 提供了提升。

### 6.3 必须避免的无效比较

不把 R332 refiner 随意接在一个不同来源 N0 的 logits 上后，直接把效果下降归因于旧 refiner。先与其原来配套的 backbone 验证保真；换 backbone 的分布变化是另一项实验因素。

不把成熟模型要求成 N0 identity。fresh refiner 检查的是“初始等于 source”；mature refiner port 检查的是“初始等于原成熟组合”。

## 7. 阶段 P2：TSRS 一项明确的小消融

新增 `H_input`：从 Round3 同一 N0 出发，`[image, seam]`，只用原 native base segmentation loss；不加 separation、preserve 或对应 auxiliary routing。原生 deep supervision 保持。

比较：

| 组别 | 输入 | 优化 | 要回答的问题 |
|---|---|---|---|
| A_R3 | image+zero | base | 继续训练参照 |
| H_input | image+seam | base | 热力图补充信息本身的增量 |
| B_R3 | image+seam | base+原V3 separation/preserve/routing | 当前局部优化组合的增量 |

`H_input − A_R3` 解释纯联合输入效果；`B_R3 − H_input` 解释**局部优化组合**效果，不能单独归因于 separation 或 routing。

优先复用 Round3 A/B。能验证相同运行代码依赖、初始化、数据序列、增强、优化器、LR、选择规则时，只跑 H_input 一个200-update run。若 reference 无法验证可比性，最多补跑一个必要 reference；没有足够 reference 时停止因果结论，不默认重跑全矩阵。

同时完成上一份 Round4 的像素变化分类：删 FP、删 TP、补 TP、添 FP，按冻结 seam 支持与其他区域统计。新报告中 gap改善95例、Recall恶化94例是现象，不预设原因是整体收缩。[S3]

结果分支：

- H_input 兼顾区域和结构，B_R3 多出 Recall 代价：优先回到局部优化组合，之后一次只动一项。
- H_input 同样有代价：先看输入图定位、归一化、先验错误与有效学习，不把问题全归给 loss。
- 200步内均几乎不变：检查样本覆盖、首层热力图权重更新和有效梯度；只能说当前短预算未形成充分证据。

R317 始终作为固定成熟参照，不重搜 α=.035；但 α 相同不代表 state attenuation 和 cap 后有效梯度相同。

## 8. 阶段 P3：RAM 只做一对“保留成熟能力”的增量试验

### 8.1 本轮推荐的唯一 RAM 新训练对照

P1 函数保真通过、O 扩展结构兼容后，运行：

| 组别 | 完整初始化 | nnU-Net输入 | 输出侧 | 训练目标 |
|---|---|---|---|---|
| M_ref_continue | 同一 R332保真组合 | image+zero | 成熟逐成员重叠模块；新增O输入为零 | 相同已核验 R332 成员/几何目标 |
| M_O | 与上完全相同 | image+zero | 同一模块；新增O输入为固定真实预测图 | 与上完全相同 |

两组均有既有的 learned overlap 处理能力。**这是“显式 O 条件是否增加价值”的对照，不是有/无骨重叠先验对照，也不能改名为 A/C。**

generator 冻结且相同。新 O 权重初始均零，两组其他权重完全相同。两组的 backbone trainability、refiner参数组、LR/优化器、采样、增强、验证、选模、步数和loss完全一致。采用本地可复现的同一有限协议；不从历史 test 结果搜索方案。

优先保留 R332 原有成员、surface/volume 约束的真实实现与既定权重；若其精确定义尚未核验，不凭摘要自编一个“surface loss”。本轮不同时换成 R350 的局部组合或额外扫描权重。

每组200 updates，两个新run合计不超过400。本比较旨在验证成熟输出模块能否吸收独立O信息，不代表完整充分训练。

无法满足结构兼容/函数保真时，不自动改跑一组新的 fresh 网络；本轮完成零更新差异与修复建议。确有代码级错误可隔离修复，但不同时搜索其它算法。

### 8.2 必须记录的机制证据

- 各组自己 backbone 输出与自己 final 输出；A 的 logits 不冒充其它组的模块前 logits。
- GT重叠区内每个成员自己的正负目标，不能把“至少2成员”广播成14通道全正。
- 逐成员 FP/FN、边界误差、错误新增成员、面积/volume变化，以及 MSD失败率和有效分母。
- O 是否响应原生/成熟模型实际犯错的位置；只凭整体生成器BCE不能证明这一点。
- base及各局部项的梯度路径、raw/weighted/attenuated/projected/capped范数、实际参数更新。
- 两步迭代分别的预测；确保 surface改善不是来源于跳过失败例。

### 8.3 解释范围

M_O 相对 M_ref_continue 的改善，才是独立 O 在成熟模块上的追加信息证据。

两者均优于 fresh V3 C，仍不能把全部差距归因于 O：成熟权重、训练历史、迭代和几何目标都不同。可作能力恢复的描述，不作单因素增益。

M_O 若没有增益或产生副作用：保留 R332，不用清空成熟 refiner 重来；先依据误差位置决定 O 条件、学习充分性或局部目标的下一项检查。不能因为显式 O 增量不成立，就把原有成员重叠先验改成一般补全。

## 9. 后续完整双先验：分阶段，不删除统一协调

本轮 TSRS H_input 与 RAM M_ref/M_O 是定位/能力保留实验，不是最终论文矩阵。后续仍要回到两数据集完整双先验。

### 9.1 RAM 成熟模型增量线

当输出侧确定了一个可保留的版本后，预先锁定它为共同 M0：

1. `M_C_continue`：无输入seam，保留相同输出重叠先验，继续训练。
2. `M_D_input`：仅增加 `[image,seam]`，其余与 M_C_continue 相同。新增 seam 通道必须在两组相同的可训练策略下能够学习；不能冻结全部输入层又期待新通道起效。
3. `M_D_local` / `M_D_coord`：固定相同双先验、相同局部loss，以预先写明的协调开关做比较。

`M_D_input` 用于隔离新增输入信息，不是删去最终 seam-local loss。完整局部loss补齐后，再比较路由，才可讨论统一机制的贡献。

R350 可提供状态、目标和参数分组的候选实现，但其历史结果是组合关联，不等于每一项已被单独证明。新实现使用 V3 修复版路由。若测试仅投影的作用，权重与cap保持相同；若整体关闭投影和cap，只能称“routing package”对照。

### 9.2 Native-start 主矩阵继续存在

等候选机制稳定后，在已登记的充分但有限预算内完成 A/B/C/D；C/D使用同一来源输出模块，B/D使用同一来源seam。依旧只训练 nnU-Net。

同样的backbone N0 + 合法预训练先验可以组成一个透明的主协议，不要求每个先验必须随机重训；同时披露所有预训练来源和成本。若使用了 mature backbone，则必须另列成熟线，不能叫native-start。

### 9.3 不能把200步变成永久闸门

本轮200步是诊断预算。既不据此判定架构已经充分收敛，也不据此保证加时必涨。

如果代码/监督/有效学习均成立，曲线显示仍在学习且没有明显质量破坏，可以**在下一轮运行前**登记所有比较组共享的更充分有限预算和资源上限；不需要先把一项极小负差包装成成功才能研究学习充分性。不允许只延长较弱arm而沿用旧A作公平对照。

## 10. TSRS 骨重叠监督：明确可执行分支

既有事实：[S1、S7] 未验证完整多成员；现有indexed/palette不能自动恢复真实多重归属。真实overlap继续unknown，不设成全零负类，也不以union、OOF FN、crowding或接触区域代替。

已知原始来源线索：`G:/gutou/标注/掌骨数据标注作业返回/flat_output_suffix`。只有本地用户已明确授权定向只读检查时才访问；未获权限写 `SOURCE_NOT_ACCESSED`，不扫整盘，不将“未访问”说成“不存在”。

- **验证到真实逐对象/多层来源**：明确完整性、成员known、有效区、与875/96的对应。只在可证明的范围生成监督；不改原标签。
- **经授权核验后仍无来源**：允许提出来源透明的 RAM→TSRS 迁移输入实验协议，但它不是TSRS-native真实重叠监督。可靠性预览、覆盖和阈值规则不等于补造GT。
- **来源未能访问**：给出精确依赖；RAM工作和TSRS seam工作继续，不重复查已完成目录。

缺少本地 overlap GT **不在逻辑上禁止**使用外部学习的 O 作为模型条件，再用真实 TSRS 分割标签训练适配；但这种做法的证据只涉及“迁移先验对TSRS分割的作用”，不能证明TSRS真实成员恢复，更不能自动满足“两个数据集各自原生建立重叠先验”的主张。

默认本轮不启动 TSRS C/D，不偷偷切换研究命名。将最终缺口写入决策表，不能用两个成熟单先验结果宣告完整双先验论文已完成。

## 11. 局部监督与统一损失的约束

继续以原生 base loss 为主体。可用概念式表达，但本地真实函数与来源才是执行依据：

`L = L_native + λ_sep L_sep + λ_ov L_member_overlap + λ_geo L_existing_geometry + λ_preserve L_preserve`。

这里 `L_existing_geometry` 只指本地已核验 R330–R332 的几何设计，不新造训练主线或自定义评价指标；具体是否合并在 `L_member_overlap` 内不重要，避免重复计权更重要。

- 二值支持与soft权重分开；非支持处权重为0。
- 标签可证明的seam背景状态与member-overlap前景状态可以分开监督；不强迫预测的S/O概率图处处互斥。
- 图中可同时有两个非空局部状态。它们在训练支持上可不相交，不表示一张图只有一种问题。
- GT/成员known/valid只进入目标与离线分析，推理不读取GT。
- preserve不能在明确需修正处同时强迫完全复制teacher；具体冲突先沿代码验证。
- 记录全参数路由及各参数组有效更新，不以“冲突点积满足契约”宣称泛化不退化。

## 12. 评价与证据层级

本轮继承 Round3 既定 evaluator、metric family、阈值及严格区域gate；旧FAIL/INCONCLUSIVE不改为PASS。

同时报告绝对差值：极小的负点估计不等于严重退化，也不自动等于浮点误差或统计非劣效。未来实际容忍范围若需要修改，另行依据用户目标预登记，不能为追认旧结果而设。

- TSRS 主看 gap/merge/边界/距离，并与Recall及像素增删分布共同解释；不得通过漏预测前景造结构改善。
- RAM 主看 overlap/pair DSC、NSD、MSD及failure、RAVD；不只挑DSC。
- R317“历史最佳”不是所有指标都占优；例如新旧Boundary IoU与全局指标并不全部同向。[S1]
- 124 test、69 val、96 val分别展示。新设计只读train/val，不新增test选择。
- 当前一组一个seed，图像/subject聚类未确定；继承[S2/S3]仅作描述的统计范围。本轮不生成p值、总体CI或“稳定提升”。`1e-4`不是任何实验都适用的稳定性阈值。
- aggregate pair指标与case平均差可能采用不同分母。先核验原聚合，不把差异直接叫算错；不从统计摘要伪造逐ID向量。

## 13. 本次授权与预算合并

执行本文件时，先检查旧Round4是否已进行。已有相同有效证据直接复用，不能把R4与R4.1各跑一遍叠加预算。

本次增量上限：

- 零更新：定向资产差异、已有checkpoint验证推理、接口保真、已计划的像素/梯度诊断。
- TSRS：H_input一个200-update run；仅在必要reference不可复用时再加一个200-update run。
- RAM：通过P1后，M_ref_continue / M_O两个200-update run。
- 因此**最多4个新增segmentation run、800 updates**；reference可复用时为3个run、600 updates。已执行的相同诊断run计入本预算。
- 不重训generator，不重训原版N0，不自动启动M_D、全矩阵、3-seed、长训、跨骨架或test。

这是对旧Round4“全轮最多400 updates”的明确修订：多出的预算仅用于新增的TSRS热力图对照及必要reference，不作为自动搜索额度。结构保真不通过的RAM分支不训练。

优化器更新之前写 `R4_1_PROTOCOL_LOCK.json`：比较问题、共享初始化、完整component hash、数据和增强序列、run源码快照、依赖/配置、参数组、预算、选模和metric gate。未来的新阶段另立协议，不以“本文件允许后续路线”为自动授权。

## 14. 最小交付，不为文件名反复审计

优先扩展已有日志；以下是需要的信息，不强迫重复生成同义文件。

1. `ANCHOR_ALIGNMENT.md`：R317/R332/R350的函数、权重、split、初始化来源和当前可比范围。
2. `R332_PORT_PARITY.json`：逐步/最终前向对齐、独立O映射及实际继承的成员/几何设计。
3. `R4_1_PROTOCOL_LOCK.json`、运行时源码归档、history、实际新增成本。
4. TSRS A/H_input/B结果及逐例/像素变化；RAM M0/M_ref/M_O结果及逐成员修正。
5. `TSRS_OVERLAP_DECISION.md`：具体分支与依据，不只写PENDING。
6. `INTEGRATION_DECISION.md`：已保住什么、损失什么、O是否被有效使用、下一步唯一最小动作。

没有跑过就写NOT_RUN。通过文件解析/哈希检查不写成“模型验收PASS”。报告中旧运行时SHA缺失仍披露，新hash不得回填冒充旧记录。

## 15. 执行结束需要回答的五个问题

1. R317/R332成熟能力在不新增训练时，是否被新接口保真保留？
2. TSRS的Recall代价更多出现于纯热力图输入，还是当前局部优化组合？证据足以支持到哪一层？
3. 独立O能否在R332成熟逐成员模块上增加有效信息，而不是只让面积DSC上涨、表面变差？
4. 哪一个成熟输出版本值得作为后续增加输入seam与统一协调的共同起点？
5. 两数据集完整双先验主张还缺什么，哪些仅是历史/探索证据？

最终目标仍然是完整双先验，不是永久停留在单先验消融；本轮先保住历史已证明有价值的能力，再公平地测新增部分。

---

## 来源与未验证项

- [S1] `2026-09-20--dual-prior--r3--round-review.md`：版本定位、精确设计和历史改善量（尤其§1–8、Next Actions）。
- [S2] `analysis-report.md`：第三轮比较问题、证据层级和决策。
- [S3] `stats-appendix.md`：配对差值、方向计数及统计限制。
- [S4] `PILOT_RESULTS(1).csv`：第三轮汇总原始数值。
- [S5] `PAIRED_DELTAS(1).json`：第三轮汇总差值及逐例统计摘要。
- [S6] `figure-catalog.md`：图目的与图注。实际PDF图片未随本次文件提供，本方案没有视觉复验它们。
- [S7] 前轮 `TSRS_OVERLAP_STATUS.md`、`RAM_IMAGE_ONLY_OVERLAP_ADAPTATION.md` 和 Round3报告：监督与先验身份。

本次没有提供R317/R332/R350原训练源码、逐例历史原始数组或本地权重内容，不能仅靠本报告确认精确loss、refiner参数键与当前环境兼容。要求Codex定向检查这些具体内容，不要求重新读GitHub。
