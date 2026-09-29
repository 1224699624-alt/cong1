# 给 Codex：基于成熟最佳实验修订第四轮，不重做项目

请在本地 `G:\gutou\YOLO+SAM` 执行。不要读取 GitHub；服务器仅沿用已经授权的运行环境与隔离目录，不扩大权限。根目录外的原始标注来源仍须已有明确只读授权。

先读相关 `AGENTS.md`、本交付包 `PLAN.md`，然后读取我刚整理的：

- `2026-09-20--dual-prior--r3--round-review.md`
- `analysis-report.md`、`stats-appendix.md`
- Round3 的结果CSV、配对JSON、协议、历史和源码记录
- 与 R317、R332、R350 有关的本地现成代码/权重/validation结果

本轮是 **V3.1 / Round4修订**。这次新上传的CSV/配对JSON与第三轮原文件相同，不是额外一次性能实验。旧Round4若已做了相同检查，直接复用，不叠加训练预算。

## 1. 主线不变

两个数据集都要走：原图＋骨缝热力图共同输入 nnU-Net → 输出骨重叠先验 → 统一局部损失/梯度协调。只用nnU-Net。

B/D输入 `[original_normalized_image, aligned_seam_heatmap]`，A/C输入 `[image, zero]`。不改回先改写图像的adapter，不用loss代替热力图输入。

输出研究对象是骨重叠和成员关系，不是普通completion/OOF纠错。TSRS重点骨缝、RAM重点重叠，不表示单张图只有一种问题。Dice/IoU守质量底线，主要结构指标要有真实收益。

## 2. 本轮优先级调整

不要继续把 fresh V3 refiner 当成唯一候选，也不要用历史分数拼一个“最佳D”。

- R317保留为TSRS成熟骨缝参照。
- R332 step2保留为RAM成熟逐成员骨重叠参照，尤其检查其两步迭代、own/other-member信息以及已有surface/volume目标。
- R350 RAM保留为统一状态/协调组合的成熟参照，不宣称routing已单独被证明。
- V3保留干净联合输入、独立O接口、正确路由和matched协议，不因新编号就替代旧最好效果。

新报告的历史结论仍要和本地实际源码、权重、split对应。只定向检查差异，不重复全盘审计。R332旧test124的数字不能直接和新val69相减；本轮不开test。

## 3. 首先完成零更新的成熟能力保真

优先复用已有可追溯validation；缺失时只做固定权重推理。

准确识别R332所对应的backbone与refiner，不把一个mature refiner随便接到不同来源N0上后，就说旧方法不行。

将原R332组合接进V3接口时，先完整保留：14个独立sigmoid成员、逐成员特征及顺序、两步迭代和feature重算、residual规则、坐标/预处理及成熟权重。

**新增输入通道可零初始化，但不能清零已训练的refiner残差层。**

禁用新增输入作用时，新wrapper输出应与原R332同输入下的每步logits和最终mask一致到预登记数值容差。先保住函数，再做训练；不能用训练掩盖移植错误。

如果只有第一层新增O特征且结构完全兼容：原特征权重严格复制，只将新O特征权重置零。若不兼容，先报告并做最小保真接口，不通过大量跳键“复用”。

继续使用现成image-only O权重：
`e7b738e0b413f9138660475062e76e5d67d9df5c94afc0a2c2597ff0cf126ae5`。
不重新训练generator。

## 4. TSRS只新增一个热力图输入对照

新增 `H_input`：同Round3 N0、同200 updates、同数据/采样/增强/optimizer/LR/trainability/选模，输入image+seam，只用原native base loss和deep supervision，不加separation/preserve及对应auxiliary routing。

比较：
- H_input − A_R3：热力图信息本身的效果。
- B_R3 − H_input：当前局部优化组合的效果，不单独归因于loss或routing。

原A/B来源可比时直接复用；必要时最多补一个reference。做先前计划的FP/TP增删定位和Recall分析，不再凭Precision/Recall推测“肯定整体收缩”。

相同alpha=.035不等于相同有效梯度，需查看state attenuation、preserve和cap后的实际更新。

## 5. RAM只做一对成熟同起点对照

保真通过后，本轮默认选择：

- `M_ref_continue`：R332保真组合＋新增O接口，实际O输入为零。
- `M_O`：同样的完整初始化和代码，实际O输入为冻结generator预测。

两组nnU-Net都输入image+zero，先不加seam。两组都保留已有逐成员重叠模块，因此这是“显式O条件是否提供追加价值”的实验，不是无先验A对重叠C。

每组200 updates，完全相同数据序列、优化器/参数组、可训练性、学习率、选模及两步迭代。保留本地已核验的R332成员及surface/volume训练目标，不同时切换R350 loss、改阈值或搜索权重。

记录各组自身backbone前后输出、逐成员正负监督、重叠区错误新增/恢复成员、surface变化、failure/有效分母，以及最终checkpoint的O使用情况。O能识别一些重叠，不表示能帮助native错误点，不能先排除这个问题。

结构保真不通过就不运行本对照，不自动换成另一个fresh大型网络。

## 6. 完整双先验仍是后续目标

成熟输出版本确定后，才用同一M0比较：成熟overlap继续训练 vs 增加输入骨缝热力图；完整局部监督接好之后，再做相同双先验/相同loss下的协调开关对照。

这不取消native A/B/C/D。两条线分表：
- native-start回答从相同无先验backbone出发的先验增量；
- mature-start回答在已有成熟能力上新增组件是否有收益。

合法预训练先验不必每次随机重训，但所有初始化、使用过的标签和预训练成本都要说明；完整mature backbone不能冒充native N0。

不要把R332整体相对R325的收益算成本轮O通道收益，也不要将R350组合收益全部算作routing收益。

## 7. TSRS overlap继续做明确决策，不改研究对象

真实multi-membership没有验证就继续unknown，不当负例、不以union/crowding/FN替代。已授权时只读已知原始来源精确目录；未授权写SOURCE_NOT_ACCESSED，不递归搜整盘。

迁移O作为TSRS输入证据在逻辑上可探索，但不是TSRS-native真实重叠监督。默认本轮不启动TSRS C/D；必须明确原生监督缺口、迁移候选和下一项依赖，不每轮只重复PENDING。

## 8. 预算、记录与结束条件

本修订最多：TSRS两run（H_input＋必要reference），RAM两run，共4×200=800新增updates；reference可复用则3×200=600。此前已经完成的同一诊断run计入，不叠加旧Round4预算。

不重训N0、不重训generator、不自动全矩阵/3seed/长训/其它骨架/test。200步只用于本轮诊断，不是最终训练充分性的标准。

更新前锁定实际run源码快照、所有依赖/配置、component权重hash、数据/增强序列、预算和选模规则。RAM历史runner SHA缺口仍披露，不能用新hash补写成旧运行时记录。

保留Round3严格gate与negative/inconclusive状态；不为涨点更改evaluator或阈值。单seed仅做描述，不生成总体p值/CI/稳定提升措辞。

最终交付：成熟资产与新实现差异、R332保真结果、TSRS A/H/B比较、RAM M0/M_ref/M_O比较、TSRS监督分支，以及下一项唯一最小整合动作。实际未运行的内容明确NOT_RUN。

**目标：不是丢掉R317/R332重新练一个弱版本，也不是把最好的历史数字拼成D；而是保留成熟能力，用可解释的接入和同起点对照，逐步完成两个数据集的双先验与统一协调。**
