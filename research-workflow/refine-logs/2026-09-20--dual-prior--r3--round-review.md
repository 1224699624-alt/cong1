---
type: results-report
date: 2026-09-20
experiment_line: dual-prior
round: 3
purpose: round-review
status: active
source_artifacts:
  - outputs/v3_round3/PILOT_RESULTS.json
  - outputs/v3_round3/PAIRED_DELTAS.json
  - outputs/v3_round3/analysis/analysis-report.md
  - outputs/v3_round3/analysis/stats-appendix.md
linked_experiments:
  - R317
  - R332
  - R350
  - R351
  - V3-Round3
linked_results:
  - outputs/analysis/latest_best_vs_baseline_20260824.csv
  - outputs/v3_round3/ROUND3_REPORT.md
---

# Dual Prior / Round 3 / Current Best Experiment Map / 2026-09-20

## Executive Summary

项目已经完成三个层级，但需要分开理解：历史成熟方法已经证明 TSRS seam prior 和 RAM overlap prior 各自有价值；V3 已把双先验工程接口清理为可解释、可隔离的实现；最新 native-start pilot 尚未证明两个先验联合后优于同预算 native continuation。

当前应保留的主版本是：

| 角色 | 当前应保留版本 | 结论 |
|---|---|---|
| TSRS 成熟单骨缝先验 | **R317 continuous α=.035** | 当前历史 seam 主锚点；结构指标强，但旧 baseline 推理路径不完全 matched |
| TSRS 最新公平单骨缝 | **V3 Round3 B** | 直接 image+heatmap，native-start matched；结构优于 A，但 Dice/IoU/Recall 略低于 A，尚未通过主张门槛 |
| RAM 成熟单骨重叠先验 | **R332 step2 iterative overlap refinement** | 当前最强 locked-test overlap 证据，继续作为最终性能锚点 |
| RAM 最新平衡损失 | **R350 epoch29** | 当前最强 validation 上 state-conditioned loss + gradient coordination 结果 |
| RAM 最新干净 overlap-only | **V3 Round3 C** | 信息隔离正确，但相对 A 区域和 surface 指标混合/负向，不能晋级 |
| 最新干净双先验+平衡损失 | **V3 Round3 D（仅 RAM）** | 工程最完整；相对 A 未通过区域 gate，重叠 surface 指标未改善，暂不晋级 |
| 历史 cross-dataset 双先验 | **R351** | 数字较好但从 R350/R332 mature 模型开始且信息耦合，只保留 optional extension |

因此目前真正适合论文核心的仍是“两条成熟单先验线”：TSRS R317 和 RAM R332。R350 RAM 可以支撑统一优化/协调的增量证据。Round3 B/C/D 是最新、最干净的因果 pilot，但不能因为版本新就替换成熟最佳结果。

## Experiment Identity and Decision Context

论文目标是：原图与骨缝热力图共同进入 nnU-Net，输出端独立使用骨重叠证据，通过局部监督和梯度协调减少冲突。当前只讨论 nnU-Net。

这一轮要回答的不是“网络能不能跑”，而是：

1. 原图+seam 的输入先验在 matched continuation 之外是否仍有增益；
2. 独立 image-only overlap generator 能否让 C/D 成为干净消融；
3. clean dual + V3 routing 是否已经达到历史 R332/R350 的效果；
4. 哪些历史路线应保留为锚点，哪些已经被降级。

## Version Map and Exact Designs

### 1. TSRS 单骨缝先验：R317，历史成熟最佳

设计版本：`nnUNetTrainerR317Continuous035`，沿用 R260 mature nnU-Net 路径。R202 一通道权重扩展成两通道，图像权重逐tensor保持，新增 seam 通道初始为0。输入为 `[normalized X-ray, aligned seam heatmap]`；R317 的 image-centers-only `RelationPriorNet(4)` 生成每图连续 seam map。分割损失之外，在 union background 的 prior 区域使用 continuous separation loss，`alpha=.035`；checkpoint 通过原始96例 validation 的 R201门槛选择。

本地模型：`outputs/nnunet/r317_image_centers_only/nnUNet_results/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_r201_selected.pth`。

R317 结果：Dice `.902858`、IoU `.826369`、Boundary IoU `.245245`、Surface Dice2 `.668408`、Gap FP `.181737`、Merge `.5000`、HD95 `14.0084 px`、ASSD `4.0475 px`。这些绝对结构指标仍强于最新 Round3 B 的多数指标。

证据边界：R317 报告中的旧 mature baseline 采用不同推理/预处理路径，不能再用它宣称严格 native gain。R317 是成熟效果锚点，不是当前最干净的 native-start 因果对照。

### 2. TSRS 最新公平单骨缝：V3 Round3 B

设计版本：`structural_prior_v3.build_backbone_input` + `nnUNetTrainerV3Round3`。A/C 输入 `[image, zero]`，B/D 输入 `[image, aligned seam]`；原图绝不由先验改写。B 从同一 `r202_two_channel_zero_prior.pth` 开始，2 epoch×100 update、seed3512、batch1、AdamW backbone lr `2e-6`。保持原 nnU-Net deep supervision/base loss；增加 separation `.035`、preserve `.10`，按有效state mass attenuation，并使用修复后的 full-parameter-space V3 routing：per-aux cap `.15`，merged cap `.15`。

结果：N0/A/B Dice `.903665/.903886/.903872`。B 相对 A 略降 Dice/IoU/Recall，但进一步改善 Boundary F1、Surface Dice、Gap FP、Merge、HD95、ASSD。它是当前最应保留的 clean seam-only 实现和实验入口，但科研结论仍为 inconclusive。

### 3. RAM 单骨缝先验：V3 Round3 B

设计版本：R325 native 的 `NnUNetMultiLabelPrior` 一通道/15 raw head严格转换为 `RamNnUNetTwoChannel` 两通道/14成员输出：复制image通道权重、新seam通道权重置0、删去第15个隐藏prior head。输入直接为 `[image, R323 RAM-native seam heatmap]`。14成员保持独立sigmoid。matched A/B均为2×100 update、seed3512、AdamW backbone lr `2e-6`，B增加separation/preserve及V3 routing。

结果：B 与 A 几乎相同；整体 DSC `+5.6e-7`，Overlap DSC/NSD和MSD轻微改善，pair指标轻微混合。它验证了 joint heatmap 输入可用，但暂时不是一个实质提升版本。

### 4. RAM 单骨重叠先验：R332，历史成熟最佳

演进链：R325 建立 native-resolution nnU-Net；R327–R329 建立共享 per-instance output refiner；R330 加入surface/volume约束；R331做五seed validation稳定性；R332联合训练多步refiner，validation选择epoch30和推理深度2。

输出refiner针对14个成员复用同一网络，输入图像、own probability、other-member max/sum、uncertainty、image gradient；研究对象是投影重叠区域的成员恢复。locked test 124例上，R332 step2 相对 R325：Overall DSC `+.000523`、NSD `+.00617`、MSD `-.17757 px`；Overlap DSC `+.003895`、NSD `+.009349`、MSD `-.1050 px`；Pair MSD `-.25676 px`。Overlap RAVD和pair MSD fail rate变差，报告中已保留。

本地 checkpoint：`outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth`。这是当前最强、最接近论文最终RAM结果的单重叠先验版本。

### 5. RAM 最新干净 overlap-only：V3 Round3 C

先用RAM真实14成员监督，将历史 image+seam generator 有限适配成 image-only generator。固定6 epoch/2550 update，选epoch5；val BCE `.055506`，precision/recall `.82174/.60363`，无塌缩。C 与 D 使用同一冻结权重 `e7b738e0...cf126ae5`。

C 的 backbone 输入 `[image, zero]`，generator 只读取image；输出 `OutputOverlapAdapter` 将 O 作为独立第7特征，不再使用 `S*(1-O)`。refiner fresh zero-residual，base、observed member-overlap BCE、preserve和V3 routing共同训练。

C 相对 A：Overall DSC `-.0000334`、IoU `-.0000659`；Overlap DSC `+.0003414`，但 NSD `-.001067`、MSD `+.001674 px`；Pair DSC `+.0003096`，但 NSD `-.001261`、MSD `+.004171 px`。说明 generator 可用，当前 output adapter/目标组合仍未把重叠证据转成一致的surface收益。C不应替换R332。

### 6. 统一状态损失/梯度协调：R350，RAM 当前最佳

R350 用统一 separation/overlap/preserve 三状态、base segmentation loss、state-conditioned auxiliary loss与parameter-space PCGrad。它不是当前最干净的“两个独立前向先验”模型，更准确的定位是**统一局部状态与协调机制实验**。

RAM validation epoch29 相对 R332 step2 anchor：Overall DSC `+.0000206`、IoU `+.0000361`；Overlap DSC `+.0007594`、NSD `+.0008252`、MSD `-.01004 px`；Pair DSC `+.0013828`、NSD `+.0015827`、MSD `-.00878 px`。相对R325 full chain变化更明显。当前它是RAM上“平衡损失/梯度协调”最好的成熟证据。

TSRS R350 相对matched native略降 Dice `-.0000175`、IoU `-.0000334`，但Boundary、Surface、Gap、Merge等均改善；相对R317总体no-go。因此 R350 不能被描述成两数据集都成功。

历史 R349/R350 使用的旧routing存在 `aux grad=None` 坐标投影缺陷；V3已修复，但这不证明历史退化由该bug导致。历史数值继续有效，未来实现不能再复制旧路由。

### 7. 历史双先验：R351，数字较好但降级为optional

R351 组合 `SeamInputAdapter`、RAM监督的 `ImageOverlapPrior`、`OutputCompletionPrior` 和局部loss/routing。TSRS从R350 mature开始且冻结backbone；RAM从R332 steps2开始。旧 generator 读取image+seam，output interaction使用 `normalize(seam)*(1-overlap)`；输入侧还先改写image再送backbone，不符合后来冻结的“image+heatmap直接联合输入”定义。

历史记录：TSRS Dice `.903889`、IoU `.828067`、Gap FP `.192697`、Merge `.572917`；RAM DSC `.979199`、Overlap MSD `1.82983`、Pair MSD `1.65415`。它们优于native参考但相对R350/R332 mature anchor存在回退，且不是native-start A/B/C/D。TSRS的overlap是RAM迁移弱证据，不是真实TSRS overlap GT。

R351 是历史上数值最好的双先验pilot之一，但不再进入core final model。保留代码与结果用于optional cross-transfer extension，不能用它覆盖Round3 clean matrix。

### 8. 最新干净双先验+平衡损失：V3 Round3 D

这是当前工程定义最完整的版本：B/D直接输入image+seam；C/D使用同一冻结image-only O；OutputOverlapAdapter独立读取O；RAM observed14-member overlap、separation、preserve进入统一局部监督；V3路由修复None坐标并增加merged cap。

RAM D相对A：Overall DSC `-.0000314`、IoU `-.0000619`；Overlap/Pair DSC略高，但Overlap NSD `-.001003`、MSD `+.002141 px`，Pair NSD `-.001142`、MSD `+.003017 px`。相对C也几乎没有额外收益。D是**最新正确实现**，不是**最新最好效果**。

TSRS D没有运行：现有TSRS标签未验证完整multi-membership；迁移weak support的阈值/可信性没有冻结。该缺口不能用union、crowding、组件接触或普通completion改名填补。

## Current Best Results Dashboard

### TSRS：当前该看哪些结果

| 用途 | 版本 | 为什么保留 | 不应怎样使用 |
|---|---|---|---|
| 成熟结构效果 | R317 α=.035 | 历史单seam结构指标最强且有完整R201 | 不用旧unmatched baseline宣称严格native增益 |
| 公平因果入口 | Round3 B | 同native起点、直接image+seam、matched A、完整96例 | 目前未超过A的Dice/IoU/Recall，不能宣称已成功 |
| 统一优化参考 | R350 TSRS | 结构指标改善，三状态/PCGrad完整 | 对R317 no-go，不作为最佳TSRS模型 |
| 双先验探索 | R351 TSRS | 有历史工程信号 | mature init、弱迁移、旧耦合；只放optional |

TSRS 当前的最高 clean global 值是Round3 A；当前 clean seam-only 的最佳结构版本是Round3 B；成熟 seam anchor仍是R317。三个角色不能合并成一行“最佳模型”。

### RAM：当前该看哪些结果

| 用途 | 版本 | 为什么保留 | 不应怎样使用 |
|---|---|---|---|
| 最强最终overlap证据 | R332 step2 | locked test上整体/重叠/surface大部分明显改善 | 不隐藏RAVD与fail-rate副作用 |
| 最强平衡损失结果 | R350 epoch29 | validation上相对R332各核心指标小幅一致改善 | 尚无对应新locked test；不是clean独立双前向先验 |
| 最新单seam | Round3 B | clean联合输入，matched A | 增益极小且混合，不作为主结果 |
| 最新clean overlap-only | Round3 C | image-only O，信息隔离成立 | 相对A区域/surface gate失败，不替代R332 |
| 最新clean dual+balance | Round3 D | 工程定义最完整 | 相对A无一致收益，暂不晋级 |
| 历史cross-transfer dual | R351 RAM | 数值较好，Pair MSD接近R350 | 从R332 mature开始，只作optional extension |

## Statistical Validation

Round3 每个arm只有一个训练seed。TSRS有96个配对case、RAM有69个配对case，但未知的患者/图像聚类使case级p值和CI不适合作为总体推断。本报告只给精确点估计、逐case均值/中位数和方向计数；没有把case SD当seed SD，也没有声称显著性。

TSRS B-A 的逐例结果见图1：结构变化较一致，但Dice 39例提高/57例下降，Recall 1例提高/94例下降。RAM 图2显示B几乎中性；C/D的DSC方向与NSD/MSD方向不一致。这些约束了“先验已经带来净收益”的说法。

## Figure-by-Figure Interpretation

### Figure 1：TSRS B 相对 A 的逐例变化

![TSRS B-A paired deltas](../../outputs/v3_round3/analysis/figures/figure-01-tsrs-b-vs-a-paired.pdf)

图存在的目的，是确认结构改进是否来自少数异常例。读者应看到Gap FP与Surface指标在多数case方向良好，但Dice和Recall分布偏负。它支持“seam监督确实在改变目标结构”，同时削弱“这个版本已获得无代价提升”的信念。下一步应把输入heatmap作用与separation loss作用拆开，而不是直接长训。

### Figure 2：RAM 各先验arm相对 A

![RAM arm deltas](../../outputs/v3_round3/analysis/figures/figure-02-ram-arms-vs-a-heatmap.pdf)

图存在的目的，是防止只选Overlap DSC这一项。B接近零；C/D在Overlap/Pair DSC上正向，却在NSD/MSD及整体区域上负向。它说明阻塞点不在generator能否识别overlap，而在output refinement和local objective是否优化了正确的surface行为。

## Eliminated or Downgraded Versions

以下版本不再作为“当前最佳”保留：

- **R333 direct dual integration**：TSRS曾使用不匹配的512 resize路径；RAM解冻后表示漂移，未通过。作为失败机制记录保留。
- **旧 final wrapper `[adapted_image,zero]`**：违背已确认的image+seam联合输入定义，被V3 joint-input替代。
- **旧 R351 clean-mainline 定位**：信息耦合、mature init、TSRS弱迁移。保留optional extension，不放主结果表。
- **R350 TSRS作为最终方法**：matched Dice/IoU略低且相对R317 no-go；只保留统一优化分析价值。
- **Round3 C/D作为RAM最佳**：工程完成但科学结果混合/负向，不能替换R332/R350。
- **TSRS C/D空行补数**：没有运行，不填0、不用RAM数字代替。
- **旧R349 routing实现用于新实验**：None坐标缺陷已确认；历史结果不删除，新实验必须使用V3修复版。
- **R349作为最新平衡损失结果**：其探索信号已被接口更完整的R350取代；仅作为R350的机制前身保留。
- **R320 U-Net结果作为当前工作项**：仍是已有跨骨架证据，但当前阶段冻结为nnU-Net，不继续维护为最新主版本。

## Where the Project Is Stuck

1. **TSRS seam存在明确结构收益与Recall代价。** B相对A没有通过区域gate。需要拆分“heatmap输入本身”与“separation loss/routing”两个贡献，确认Recall下降来自哪里。
2. **RAM overlap generator已经可用，refiner目标仍不对齐。** C/D提高Overlap DSC但恶化NSD/MSD，表明当前BCE式member-overlap局部项或fresh refiner更偏区域填充，未恢复surface质量。
3. **TSRS overlap监督仍未知。** 没有可信multi-membership或预注册weak support时，完整TSRS dual不能进入core。
4. **协调机制没有直接对照。** Round3 D未超过A；D-no-routing尚未运行。现在直接证明routing价值既不充分，也未必值得优先扩展。
5. **证据仍是单seed。** 任何小于约1e-4的差值都可能不稳定；不能进入“稳定提升”措辞。
6. **RAM pilot存在entrypoint SHA记录缺口。** 数值与协议/权重SHA绑定，但runtime runner SHA为事后重构，见`RAM_RUNNER_PROVENANCE_NOTE.json`。

## What Changed Our Belief

- 加强：直接image+seam输入是正确且可学习的工程路线；TSRS seam信息能持续改善gap/boundary/component方向。
- 加强：RAM image-only overlap generator可以在真实成员监督下独立训练，C组信息隔离问题已经解决。
- 削弱：只要拥有更好的overlap概率图，输出分割surface就会自然改善。Round3 C/D不支持这一点。
- 削弱：dual + routing 在短预算下必然超过单prior或matched continue。当前证据为负向/混合。
- 未解决：TSRS如何获得可辩护的overlap监督；routing在最终方法中的独立贡献。

## Next Actions

1. TSRS做一个预注册的小消融：`image+seam, base loss only` 对 `image+seam + separation/preserve/routing`。保持Round3 A/B的所有其他设置，定位Recall代价来源。没有该对照前不进入3-seed。
2. RAM优先做output侧目标诊断，而不是增加epoch：检查C/D逐例中Overlap DSC提高但MSD恶化的区域，比较fresh V3 refiner与R332 mature per-instance refiner的残差/边界行为。下一轮只允许一个冻结方案。
3. 若选择继续RAM clean dual，先预注册 `C_base_loss` 或同源mature-refiner初始化对照；D-no-routing应等到D本身值得保留后再做。
4. TSRS C/D继续blocked，直到原始标注成员信息或透明的weak-evidence协议完成。
5. 只有一个配置在matched区域gate和主要结构家族上同时成立后，再运行3 seeds、长训或locked test。

## Artifact and Reproducibility Index

- 最新总报告：`outputs/v3_round3/ROUND3_REPORT.md`
- 机器可读结果：`outputs/v3_round3/PILOT_RESULTS.csv/json`
- 配对差异：`outputs/v3_round3/PAIRED_DELTAS.json`
- Round3协议：`outputs/v3_round3/ROUND3_PROTOCOL_LOCK.json`
- 先验适配：`outputs/v3_round3/RAM_IMAGE_ONLY_OVERLAP_ADAPTATION.md`
- TSRS监督状态：`outputs/v3_round3/TSRS_OVERLAP_STATUS.md`
- C/D smoke：`outputs/v3_round3/C_D_REAL_BATCH_ACCEPTANCE.json`
- 严格分析：`outputs/v3_round3/analysis/analysis-report.md`
- 统计边界：`outputs/v3_round3/analysis/stats-appendix.md`
- 图目录：`outputs/v3_round3/analysis/figures/`
- 历史TSRS表：`research-workflow/refine-logs/R320_PAPER_READY_EXPERIMENT_TABLES.md`
- 历史RAM locked test：`research-workflow/refine-logs/R332_RAM_LOCKED_TEST_VS_R325_RESULT.md`
- R350修正表：`outputs/analysis/latest_best_vs_baseline_20260824.csv`
- R351定位：`research-workflow/refine-logs/R351_TRACKER.md`

本项目未绑定本轮使用的Obsidian写回目标，因此报告保存在仓库`research-workflow/refine-logs/`，没有写入外部vault。
