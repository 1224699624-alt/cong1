---
type: results-report
date: 2026-09-27
experiment_line: ram-overlap
round: r00
purpose: web-analysis-handoff
status: complete
source_artifacts:
  - outputs/ram_source_correctability_20260924/DIAGNOSIS_AND_ACTION.md
  - outputs/ram_o_source_head_20260924/DIAGNOSIS_AND_ACTION.md
  - outputs/ram_intensity_hypothesis_20260927/POSITION_AND_PRIOR_DECISION.md
  - outputs/ram_overlap_fn_geometry_20260927/DIAGNOSIS_AND_NEXT_ACTION.md
  - outputs/ram_original_paper_audit_20260927/RAM_W600_ORIGINAL_PAPER_EVIDENCE.md
linked_experiments:
  - research-wiki/experiments/ram_source_correctability_20260924.md
  - research-wiki/experiments/ram_o_source_head_20260924.md
  - research-wiki/experiments/ram_overlap_fn_geometry_20260927.md
linked_results: []
---

# RAM overlap / Cross-round（r00）/ 网页端分析交接 / 2026-09-27

> 这是一份跨轮内部交接，不是论文结果段落；r00 只表示跨轮汇总，不是新增实验轮次。此文件没有启动训练，也没有读取本地 test。RAM 是主数据集，TSRS 是额外数据集，本总结的因果表不混入 TSRS。

## 1. 当前最重要的结论

RAM 的主要困难是**真实重叠处第二个正确骨成员被强烈压成负类，尤其集中在该成员的边界附近**，不是整块骨通常都没被看到。成熟 nnU-Net/R332 在总体分割已经很强，但显式 O 先验在这些真正困难位置覆盖很弱；旧接入通道的权重为零，后来让 O 真正接入 source head 的同起点对照也没有带来分割收益。简单“更白的地方就是两骨重叠”只在总体均值上成立，对这批困难漏检无效。当前不能把输入端、输出端或任何新模块宣称为有效修复，也不能从离线理想修复量推算实际可达提升。

## 2. 数据与评价口径

- 本地 RAM-W600：train425 / val69 / test124。近期全部新增诊断和训练只用 train425、val69；未使用 test，也没有运行 TSRS。14 个独立二值骨成员可在同一像素同时为真；预测用独立 sigmoid，不可改成互斥 softmax。
- 核心同起点实验从 mature R332 继续后的 fresh100（C0）出发，模型 SHA256：ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384；恢复配套 AdamW/scaler/step100。标准 RAM val69 指标沿用原 R329 evaluator：Overall / Overlap / Pair 的 Dice、NSD@2px、MSD 等。C0 与追加384更新的结果属于不同预算；因果解释主要看同预算组间差。
- 论文 [RAM-W600 arXiv v3](https://arxiv.org/html/2507.05193v3) 的 benchmark 是 512×512、test124、监督模型5 seeds/100 epochs，并没有 nnU-Net 或我们的 O/seam 先验。本地原生分辨率 val69 的数不能与论文 test 表直接相减。论文模型家族的相对趋势只用于设计假设。

## 3. 标准验证结果：哪些是模型收益，哪些不是

以下均是 RAM val69 最终两步输出，指标为0–1值；OHEAD_ZERO 与 F0S1 原有模型参数及16项指标逐位一致，故省去重复一行。

| 条件 | 新增更新 | Overall Dice | Overlap Dice | Pair Dice | Overlap NSD@2px | 固定H0真恢复 |
|---|---:|---:|---:|---:|---:|---:|
| C0 mature fresh100 参考 | 0 | 0.979408 | 0.877243 | 0.835884 | 0.788410 | 0（H0定义起点） |
| F0S0：原R332目标继续 | 384 | 0.979349 | 0.877187 | 0.835152 | 0.788287 | 8/71,335 |
| F0S1：加固定H0 source正监督 | 384 | 0.979349 | **0.877535** | 0.835468 | 0.787579 | 36/71,335 |
| F1S0：额外解冻D2、原目标 | 384 | 0.979254 | 0.876616 | 0.834955 | 0.787000 | 20/71,335 |
| F1S1：额外解冻D2＋source监督 | 384 | 0.979248 | 0.877128 | 0.835333 | 0.786555 | 62/71,335 |
| FIN_F0S1：正监督改到最终logit | 384 | 0.979356 | 0.877496 | 0.835470 | 0.788246 | 13/71,335 |
| OHEAD_REAL：同F0S1，source接真实O | 384 | 0.979347 | 0.877533 | 0.835458 | 0.787579 | 36/71,335 |

F0S1 相对同预算 F0S0 的 Overlap Dice 仅增加0.0348个百分点，同时 Overlap NSD 下降0.0708个百分点；它只对冻结C0多0.0292个百分点的Overlap Dice，Overall/Pair/边界更低。F1 开放更多 source 路径、final-logit监督、真实O source-head 都没有得到综合标准收益。负类保护的另一次100步对照减少FP，却增加FN并降低Overlap/Pair指标；850步的身份分支真实身份优于打乱身份仅约0.0027个百分点Overlap Dice，且两组均未超过共同成熟起点。不能把这些失败分支列为“最新最好模型”。

## 4. 漏检的真实机会与白盒机制

- C0 val69 真实 overlap 正成员条件漏分率174,392/2,278,694=7.65%；非重叠正成员为81,958/11,903,740=0.689%，约差11.1倍。固定难例H0为71,335个成员错误、70,751个唯一像素。在保存的预测副本上只理想修好H0且无新FP，Overlap Dice 0.877243→0.906992（+2.9749个百分点）。**这是 oracle 上界诊断，不是任何模型结果。**
- 近期新增只读几何审计：完整真实重叠正成员中约9.44%距自身GT边界约1px；已有H0匹配审计抽到的32,732个val难例中该比例44.82%。严格子集10,265个位置里，10,249个已经预测出恰好一块骨；漏掉成员的source/最终概率中位数为0.12%/0.76%。这些子集有选择偏差，不能直接推广到全部H0，但明确指向“第二成员的边界延续”问题。
- R332 refiner每步logit修正绝对值≤2.5，两步累计≤5；C0真实重叠成员FN中约40.9%的source低于−5，固定source的末端修正不可能翻成正类。source监督在真实AdamW更新下让H0平均source logit约−7.387→−7.081、最终logit约−5.573→−5.372，仍远离0，只多修28个H0。共享decoder的正向推动更易让接近0的GT负成员跨阈值成为FP。梯度非零、参数更新和跨过−5均不能冒充最终成员恢复。
- O生成器用image-only、192分辨率、全图overlap BCE训练；完整val的重叠检测在0.5下recall约60%，但固定H0唯一像素达到0.5仅15.07%。O仅表示“可能有重叠”，没有成员身份。原C0第二输入和refiner新增O通道的首层权重均严格全零，直接放O进去无效果。新source-head O臂384步确有梯度与更新，但相对零O组H0仍36对36，额外FP+40、标准指标微降。
- 图像亮度审计：val69完整真实overlap/单骨/背景平均8-bit灰度为206.61/185.04/74.50，69/69病例overlap均值更亮；但train锁定灰度阈值234在val只召回9.54% overlap、误标399,509个单骨像素。严格H0/单骨匹配位置raw亮度AUC约0.501（因匹配已控制亮度），local-z正向AUC约0.370；不能用全图亮度二值化补难例。

这些证据证明了旧O入口无功能、先验难例覆盖低、当前监督/输出修正交付不足；**没有证明唯一主因**是网络容量、学习率、训练不足、标签错误或X光不可辨。原图—实际输入—标注的医学人工复核仍未完成。

## 5. 把O放输入端：可能性与限制

输入端若能被学习，确实可在source形成前影响多尺度特征，从机制上比固定source后的限幅refiner更有机会补强负成员。但当前C0第一卷积第二输入权重全零、encoder首层冻结，所以简单把[image,zero]换成[image,O]在现有checkpoint上是**严格无效**。旧R351同时改写image并结合seam/输出prior；Round3 B/D输入的是seam，不是干净的“仅O换位置”对照。

若以后给输入位置一次小型验证，应使用相同C0、相同新增可训练零初始化O入口的OZERO/OREAL两臂，只改变O=0或冻结真实O；训练序列、optimizer/LR、R332目标和val69评价一致，并追踪早期stem→高分辨率skip→source→两步最终预测。O在H0信号弱且上次source-head试验为负，故**此对照目前NOT_RUN，也不应直接长训**。

## 6. RAM原论文模型差异和可借鉴之处

原论文有13个监督模型（UNet、DeepLabV3、FPN、PSPNet、DeepLabV3+、SegResNet、UNet++、SegFormer、TransUNet、UKAN、UMambaBot、UMambaEnc、SwinUMamba）和SAM box/point、MedSAM box；**没有nnU-Net**。下列总体指标是原论文Table4 test均值；“重叠14骨对平均”是对论文Table12/13逐对数字的算术平均，非论文另报总分。

| 模型 | 总体Dice% | 总体NSD% | 14骨对平均Dice% | 14骨对平均NSD% |
|---|---:|---:|---:|---:|
| UNet | 96.79 | 83.36 | 65.80 | 66.61 |
| UNet++ | 97.33 | 86.99 | 70.16 | 71.25 |
| TransUNet | 97.62 | 89.48 | 73.27 | 75.66 |
| UMambaBot | 97.53 | 88.89 | 72.70 | 74.64 |
| UMambaEnc | 97.56 | 89.10 | 72.45 | 74.73 |
| SwinUMamba | **97.75** | **90.71** | **74.45** | **77.16** |

论文中SwinUMamba相对UNet总体Dice只高0.96个百分点，14骨对平均Dice高8.65个百分点；例如Cap–Tz骨对Dice为60.01%对49.31%。这说明不同模型的主要差距更集中在重叠骨对，而非总体前景。它**不能单独证明注意力、Mamba或skip哪一个模块是原因**，因为容量和结构同时变了。对当前nnU-Net，更值得借鉴高分辨率/多尺度skip、边缘保留和远距离骨形状上下文的小模块，而不是直接换SwinUMamba或SAM骨架。

## 7. 统计边界、失败版本和当前决策

近期正式对照多数是同一seed、短程384步和val69，不能宣称已训练充分或多seed稳健。严格H0匹配只覆盖一部分难例；诊断AUC/概率读出不是分割成绩。C0零新增更新与384步组不同预算；论文test124与本地val69/不同分辨率更不能直接相减。没有新增图，所有本总结的表和计数均可由下列JSON/CSV复核。

当前应停止推广：简单白度阈值、新增显式O直接输出/source-head、单纯source或final H0正监督、D2解冻、负类保护、现有身份分支。成熟R332/C0只作为能力锚点和下一次共同起点，不把局部Overlap Dice微高的F0S1冒充综合最优。TSRS R317仍是另一数据集的历史骨缝锚点，未参与本轮RAM因果表；R351只保留optional extension，不能回流为core final。

**唯一最小下一步：**冻结C0，在train425的固定H0与同病例/同成员匹配真负位置提取高分辨率decoder/skip特征，只在train拟合小型成员条件读出，在val69检查困难第二成员能否被区分。若特征已有明显且病例稳定的条件信息，才锁定一个保留R332成熟能力的成员条件高分辨率修正模块，同起点、同预算对照，并同时要求H0恢复与Overall/Overlap/Pair Dice、NSD改善且FP受控。若读出不通过，先做盲态图像/标签复核并改进信息来源，不继续盲换先验位置。该读出与后续训练目前均NOT_RUN。

给网页端分析时，请重点评估：（1）现有证据更支持表征不足还是监督/参数更新未把已有信息转成正确第二成员？（2）冻结特征读出应怎样设置最小且防GT泄漏的H0/负例对照？（3）若读出通过，最小成员条件修正点应选高分辨率skip、decoder head还是早期O输入，如何只变一个因素并保护FP/NSD？请不要把oracle +2.97pp、诊断AUC或论文test排名当成本项目实际增益。

## 8. 证据和复现索引

- C0及白盒：outputs/whitebox_adam_20260922/fresh_adam/FINAL_VALIDATION.json；outputs/ram_feature_probe_20260922/DIAGNOSIS_AND_ACTION.md。
- H0与2×2：outputs/ram_source_correctability_20260924/DIAGNOSIS_AND_ACTION.md、v4/FACTORIAL_RESULTS.csv、v4各臂FINAL_VALIDATION.json和TRAINING_TRACE.jsonl。
- final监督：outputs/ram_h0_final_target_20260924/DIAGNOSIS_AND_ACTION.md。
- O信息/零权重/接入：outputs/ram_prior_info_20260924/READOUT_CROSSCHECK_RESULT.json、C0_PRIOR_INGRESS_WEIGHT_AUDIT.json；outputs/ram_o_source_head_20260924/DIAGNOSIS_AND_ACTION.md、OHEAD_ZERO_PARITY.json、PAIR_SUMMARY.json、RESPONSE_COMPARISON.json。
- 灰度与边界：outputs/ram_intensity_hypothesis_20260927/analysis_v3/INTENSITY_AUDIT_SUMMARY.md；outputs/ram_overlap_fn_geometry_20260927/DIAGNOSIS_AND_NEXT_ACTION.md、INPUT_O_FEASIBILITY.md。
- 原论文：outputs/ram_original_paper_audit_20260927/RAM_W600_ORIGINAL_PAPER_EVIDENCE.md；[RAM-W600 arXiv v3](https://arxiv.org/html/2507.05193v3)。

本仓库未识别为已绑定的Obsidian项目，本报告保存为本地Markdown；没有执行Obsidian写回。
