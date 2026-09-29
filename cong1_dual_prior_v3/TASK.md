# TASK V3 — TSRS / RAM 骨缝—骨重叠双先验
## nnU-Net 完整接入、局部监督与统一协调执行规格

**编制日期：2026-09-19。不是训练完成报告。当前只推进 nnU-Net。**

> 对两个数据集分别建立/适配骨缝先验和骨重叠先验。骨缝先验作用于通用分割模型输入端，骨重叠先验作用于输出端。统一局部损失与梯度协调减轻二者的优化冲突。Dice/IoU 守住基线，TSRS 优先改善骨缝结构指标，RAM 优先改善骨重叠指标。

## 0. 依据优先级与禁区

用户本轮明确目标优先于此前任务书、V2/V2.1 和历史实验措辞。先读项目 `AGENTS.md`，再读本文件、`REPO_AUDIT.md`、`SOURCE_INDEX.json`、R351 plan/review/tracker/resume。

本版本撤回旧包的以下要求：将输出先验定义成一般 completion；把 OOF-FN×crowding 作为必需核心；强制五折 OOF；只补 TSRS 不检查 RAM；当前扩张 TransUNet/SwinUMamba；把结构指标全部降成可有可无的附录。

禁止：重训原生基线作为默认第一步；改变原始数据/split；删除历史资产；绕过 checkpoint 来源核验；test 调参；无界搜索；把未知重叠当负标签；把一般漏分/拥挤/合成遮挡直接当真实骨重叠 GT；声称损失数学保证不退化。

`completion/refiner` 可以继续作为历史类名或重叠先验执行修正的动作；不能取代“骨重叠”这个研究对象。

## 1. 研究边界与完成状态

### 1.1 两个数据集都启用两个先验

| 数据集 | 输入端 | 输出端 | 主要结构目标 | 次要结构 |
|---|---|---|---|---|
| TSRS | 骨缝先验 | 骨重叠先验引导修正 | gapFP、merge 及边界/拓扑证据 | 重叠相关行为；真实 GT 不足时说明不可直接验证 |
| RAM | 骨缝先验 | 骨重叠先验引导成员修正 | overlap/pair 的形状与表面指标 | 固定 seam 区域误报、分离保真 |

TSRS 偏骨缝、RAM 偏重叠是用户的研究设定，不是本任务已测得的精确占比。不得写成 TSRS 没有 overlap、RAM 没有 seam，也不得以数据集名字为依据把次要先验永久关闭。单图可同时有两类区域。

### 1.2 三个完成层级必须分开

- **工程接入完成：**输入/输出分支独立、可组合，真实 batch 可训练/推理，无 GT 泄漏。
- **监督证据完成：**四个先验各有来源、可观测性、可信度和适用边界；TSRS weak/unknown 不能算真实重叠监督已补齐。
- **实验结论完成：**同一起点和协议的对照支持区域保真与结构收益；当前尚无新实验结果。

一个层级通过不能冒充另外两个层级通过。

## 2. 首先复用已有无先验权重

仓库 tracker 已记录如下资产，先定位并验证，不默认重新训练。[S04]

| 数据集 | 查找尾路径 | 预期历史 SHA256 |
|---|---|---|
| TSRS | `r317_image_centers_only/r202_two_channel_zero_prior.pth` | `55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b` |
| RAM | `r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth` | `88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9` |

只在用户已授权的本地/服务器环境搜索；不要把旧记录的 IP、路径、GPU、磁盘余量当当前事实。`.gitignore` 排除了大权重和数据，GitHub 可见日志不证明运行时文件还在。[S17]

输出 `ASSET_LOCK.json`，逐项记录实际路径、文件大小、SHA256、checkpoint keys、模型结构、输入/输出语义、初始化祖先、训练 split、预处理、选择规则和 prior-free 证据。

TSRS 必查：两通道转换是否与原 R202 数值等价；第二通道是零占位还是已训练 seam。RAM 必查：14 独立 sigmoid 成员、15 通道 raw head 的用途；plain 路径没有启用 prior residual/结构辅助。[S18,S19]

原版官方 pipeline 与仓库的 nnU-Net 多标签适配须准确命名；不得为了命名差异重训。若用户本地有更明确的原版权重，核验后优先使用，并保留与 tracker 资产的对应关系。

**N0 = 已有无先验模型，直接重新验证；A = 同一起点继续训练的公平对照。A 不是从头训练 N0。**

R317/R323 先验生成资产、R332/R350 成熟模型、R351 旧 pilot 全保留。成熟模型不作为新 A/B/C/D 的共享原生起点；旧结果单列 mature reference。

## 3. 总体接口：位置清楚、信息可独立隔离

核心前向：

```text
S_d = P_seam^d(X)
O_d = P_overlap^d(X)                 # 干净核心要求不依赖有效 seam 信息
X'  = InputSeamAdapter(X, S_d)
z   = BackboneContract_d(X')        # 当前固定为 nnU-Net
z*  = OutputOverlapAdapter(X, z, O_d)
pred = NativeOutputContract_d(z*)
```

统一损失只存在于训练阶段；推理使用学到的参数和图像推导先验，不计算 GT loss。零初始化/限幅等前向保护需在训练与推理一致使用。

### 3.1 输入骨缝分支

优先保留 R317（TSRS）/R323（RAM）生成器和已验证协议；接入为输入残差或经证明等价的输入适配。不得访问 backbone 内部特征名。

新 clean core 不默认让 overlap 进入输入 adapter。旧 `SeamInputAdapter(X,S,O)` 保留 legacy 复现；新分支将信息依赖写明。不要同时改多个耦合方式后无法解释归因。

已有 TSRS 两通道 native 若语义为 `[image, zero]`，新核心可保持此契约，输入适配后的图像与原 zero 占位送入 backbone。向占位通道直接填 seam 属于另一接入设计，不能混入 A/C，也不能直接继承历史成熟权重收益。两种接法择一冻结，不做无界搜索。

### 3.2 输出骨重叠分支

保留 `ImageOverlapPrior` 的语义：输出反映重叠关系/概率的证据。`OutputCompletionPrior` 只作为可复用的执行修正模块，建议新接口名 `OutputOverlapAdapter`。

至少显式输入 `image, logits, overlap_probability`，RAM 可沿用 own/other-member/uncertainty/gradient 特征。**overlap 必须有独立信息通路，不能只作为 `S*(1-O)` 的因子。** 首版不增加新的 learned crowding head、实例语义分类器或大型骨架。

native-start 新输出残差末层零初始化、残差有界、初始输出等于 native。历史 RAM refiner 可作为带来源的预训练初始化选项，C/D 必须同源同样加载；不默认加载已联合改变 backbone 的 R332 state。使用迁移/预训练须报告额外数据与成本，不能称 fresh。

### 3.3 overlap generator 与干净消融

现有 `ImageOverlapPrior(X,S)` 读取 seam。[S08] 不能训练时给 S、C 组推理时置零就声称“仅重叠先验”验证完成。

优先在原有小型先验网络上建立 image-only 变体：可迁移兼容层，明确记录第一层通道处理，在 RAM train/val 上按同一真实重叠定义做有界适配/校验。只改变先验信息依赖，不改变分割骨架。源旧模型不可覆盖。

若尚未完成 image-only 适配，允许先复现旧 mixed-context pilot，必须标记 `LEGACY_COUPLED_CONTEXT`，不能进入 clean A/B/C/D 主表。不得用此临时路径阻止其他独立审计继续完成。

必要的 image-only 先验适配单列为 M1 内的有界子任务：仅 RAM train 更新小型 overlap generator、RAM val 按预登记 BCE 选择，拟定上限可沿用历史 6 epoch 量级；这不是重训 nnU-Net。先登记额外成本、冻结预算并显式启用 `prior_adaptation.enabled`，不可把它隐藏在“零成本复用”中。之后 C/D 使用同一冻结版本，TSRS 迁移继续标 weak。

### 3.4 模型契约保持

TSRS 二分类 foreground margin 可采用 `z_fg-z_bg`，要测试其 sigmoid 与原 softmax foreground 等价；RAM 14 成员独立 sigmoid，绝不能改成互斥 softmax。保持原生 deep supervision、滑窗、TTA、padding/回采样和输出坐标。

所有先验关闭时，像素输出、阈值 mask、主指标必须复现 N0，差异超出已冻结数值容差则停止该分支。

## 4. 四个先验的来源与 TSRS 证据缺口

每个数据集都输出一份 `PRIOR_SEMANTICS_<dataset>.json`：source dataset、source labels、训练/验证来源、checkpoint SHA、是否 frozen、图像预处理、输出含义、可观测性、可信度、限制。

### 4.1 RAM

seam：R323 RAM-native 资产，核验生成链与 train/val coverage。

overlap：优先复用 R351 RAM 多成员监督的先验；真实标签为同一像素至少两个已确认骨成员。R327–R332 的逐成员修正/迭代设计可复用，不将历史 joint checkpoint 当原生模型。

### 4.2 TSRS

seam：保留 R317。

overlap：先检查本地是否还有逐骨 mask、多层 annotation、原始标注文件，而不只检查已经转成二值/索引 PNG 的导出。审计允许改变读取方式，不允许凭空恢复被丢失的多成员信息。

若确有可信多成员标注，可建立 TSRS 原生重叠监督；如果只有现有互斥颜色/索引标签，则真实 overlap 状态为 unknown。匿名实例能辅助核验骨缝/几何，但不能通过颜色相邻、膨胀相交、FN 或 crowding 推出真实重叠。

现有 RAM→TSRS 迁移重叠先验允许作为**明确标注来源的弱证据和工程 pilot**，不再一概禁止 RAM 权重。可在 TSRS train 上训练输出适配器，使真实二值分割目标在该弱证据下得到保真/修正。准确名称为“TSRS 上的重叠弱先验引导适配”，不是“TSRS 原生真实重叠监督”。

用户当前要求针对两套数据分别建立两个先验；不能把共享一个 RAM teacher 自动写成四个数据集原生先验都已建立。TSRS 的输出适配器在 TSRS 上训练，也不等于 overlap generator 已获得 TSRS 原生重叠监督。若原始标签审计仍不支持这种监督，标记 `NATIVE_OVERLAP_SUPERVISION_GAP`，列出确切缺少的标注和已检查的来源，保留为尚未满足的研究目标；不能用 general completion 掩盖。明确标注弱迁移的工程 pilot 可继续，但不能冒充最终监督方案已经定案。

本轮不强制增加 synthetic 重叠生成、大规模重标注、OOF 五折训练或新课题。只在新的原生监督方案被单独冻结后再实施；合成重叠至多证明构造样本的监督，不等于真实 TSRS 投影重叠。

## 5. 局部监督契约：不按图片/数据集二选一

所有输入对齐到同一空间坐标。支持和权重分别保存：支持为 bool，权重为 float32 且为零的地方不参与局部损失。

### 5.1 基础对象

- `V_spatial`：真实图像/有效标签区域，排除 padding/ignore。
- `U_gt, K_union`：union foreground 与其可观测性。
- `Y_k, K_member_k`：骨成员标签及逐像素逐成员可观测性；TSRS union channel 不是完整成员列表。
- `O_gt, K_overlap`：可观测的真实 overlap 标签及有效性。
- `S`：图像推导 seam；`O_teacher`：冻结的图像推导 overlap 证据。

RAM 全成员完整可观测时：`U_gt=any(Y_k)`、`O_gt=(sum(Y_k)>=2)`。部分成员未知时，看到两个肯定阳性可证明 overlap 阳性；只有一个/零个已知阳性，不能未经完整性证明认定 overlap 阴性。TSRS 二值前景可观测，不意味着成员 overlap 可观测。

### 5.2 可信骨缝状态

```text
I_seam = V_spatial & K_union & (~U_gt) & seam_support(S)
W_seam = float(I_seam) * seam_confidence(S)
```

复用已冻结的 seam support 规则，R351 legacy 为 normalize(S)>=0.25，仅作可追溯起点；不得根据 test 重搜。可信骨缝只选 union background，不能把重叠前景误作“应挖掉的缝”。

### 5.3 可观测真实骨重叠状态

```text
I_overlap_obs = V_spatial & K_overlap & O_gt
W_member = float(I_overlap_obs) * K_member * member_confidence
```

输出侧在该区域监督成员，优先使用已观察到的正/负成员标签的局部 BCE，不能仅靠“往所有通道加前景”提高重叠。可保留旧正成员 softplus 作为候选，但应由同一完整分割损失约束非成员假阳性，记录选择。

RAM 已有固定 train pair list 时，可复用局部 pair-intersection 辅助；不额外搜索新的关系图。pair loss 不是两数据集必须同样具有的 GT 项。

### 5.4 TSRS 未观测 overlap 下的弱投影监督

真实成员 overlap loss 不可用，不伪造；但输出 overlap 分支不因此删除。

可选且需预注册的弱局部支持：

```text
I_overlap_weak = V_spatial & K_union & U_gt & (~K_overlap)
                 & trusted_support(O_teacher)
W_overlap_weak = float(I_overlap_weak) * confidence(O_teacher)
```

在这个区域使用**真实可观测 union GT**监督最终输出，名称为 `overlap_conditioned_union_loss`。O_teacher 只决定弱支持/权重，不成为真实成员 GT；不使用 OOF FN 定义该状态。teacher 冻结、support detach、阈值/可信度规则冻结，不随当前学生预测移动。

`trusted_support` 必须有来源记录和覆盖审计；没有可信度依据时可暂时不启用弱局部项，仅保留 overlap 前向条件和全图真实 segmentation loss，状态写 `WEAK_LOCAL_TERM_NOT_VALIDATED`。不能用“加了 loss 名字”冒充监督补齐。

TSRS weak overlap 的空间覆盖率只叫 predicted/weak support mass，不能当真实重叠占比，也不能直接评价真实重叠精度。

### 5.5 保真与状态关系

```text
I_overlap_active = I_overlap_obs | I_overlap_weak
I_preserve = V_spatial & K_union & (~I_seam) & (~I_overlap_active)
```

当前 arm 关闭哪一项，就不保留它的训练状态来偷偷改变 loss；例如 B 不构造有效 overlap 局部项，C 不构造有效 seam 局部项。

assert：W 的非零支持属于对应 I；I 在 valid 内；I_seam 与 overlap_active 不相交；有效 union 监督区域按已启用状态正确分区。空支持返回与 logits 连接的有限零损失。

同一 batch/单图可同时包含非零 seam mass 和 overlap mass。这个监督分区不意味着两种真实问题按图互斥。原始 S/O 同时高时只记录 prior conflict，不能让原始概率图被强制清零自证无冲突。

## 6. 统一损失：同一结构，不伪造同一监督强度

```text
L = L_seg_native
    + lambda_s * L_seam
    + lambda_o * (L_overlap_observed + beta_weak * L_overlap_conditioned_union)
    + lambda_p * L_preserve
```

- `L_seg_native`：同一数据集的 A/B/C/D 完全相同，主监督作用于最终输出。TSRS/RAM 的标签表示不同，base loss 的具体实现可不同。
- `L_seam`：只在可信背景缝隙抑制 union foreground。
- `L_overlap_observed`：可观测的重叠成员约束；无可观测标签时返回有标记的 unavailable/零贡献。
- weak union 项：是真实 union 监督的结构加权，不是真实 overlap 检测损失。
- `L_preserve`：非干预区域保持与共同冻结 native teacher 的分布一致。teacher 不由每组自己滚动更新，也不偷偷用成熟先验模型替代。

所有局部项使用 active-mass normalization；面积很小、可信度低的区域用同一显式 attenuation 规则控制全局强度。若复用 R350 mass_emphasis，应排除 padding/unknown 后计算，不直接对全 padded tensor 求 mean；实际调用与日志必须对应。

少量与大量结构状态可以自然产生不同有效权重，但不能用 `dataset==TSRS` 把 overlap 强行设为 0，也不能用 `dataset==RAM` 把 seam 设为 0。缺标签导致的 unavailable 与按数据集取消机制是两回事。

## 7. 梯度协调修复与验证

复用 R349 的参数空间思想，但不能原样复制其 None 边界缺陷。先运行随附探针，再改隔离版本或以最小、可回退 diff 修复，并跑旧回归。

对同一 trainable parameter list，将缺失的 auxiliary 坐标视为 0；全空间投影：

```text
g_j' = g_j - min(dot(g_j,g_base), 0)/(||g_base||² + eps) * g_base
```

仅 base 和 auxiliary 都不使用的参数才可保持 None。base norm 近零要有明确策略（默认不放大结构辅助），空 loss/无 grad/weight=0 需安全跳过。负的请求权重应拒绝。

既记录每项 cap，也检查合成辅助范数；不把每项 `.15` 表述为合计 `.15`。如新增总 cap，应记录为新版本规则，不混入历史结果。

日志至少记录 g_base norm、各原始/投影/应用 norm、投影前后 cosine、seam-overlap cosine、最终合成 auxiliary/base ratio、可观测与弱监督的有效质量。参数空间路由只是一阶方向控制；Adam 动量、离散步长、泛化变化仍可能导致退化。

**不得宣称梯度投影自动保证验证 Dice/IoU 不下降。** 最终靠独立评价 gate 和对照判断。

## 8. N0 + A/B/C/D + 协调对照

| Arm | 起点 | 输入骨缝 | 输出骨重叠 | 训练说明 |
|---|---|---:|---:|---|
| N0 | 已有 native | × | × | 不新增训练，只复现推理基线 |
| A | 同一 native | × | × | matched continuation，base loss |
| B | 同一 native | ✓ | × | seam-only，局部 seam + preserve |
| C | 同一 native | × | ✓ | overlap-only，可观测/弱 overlap + preserve |
| D | 同一 native | ✓ | ✓ | 完整双先验与统一协调 |
| D_no_routing | 与 D 相同 | ✓ | ✓ | 同一局部项与权重，只关闭梯度路由 |

A/B/C/D 两个数据集都跑。D_no_routing 在主四组管线通过后做，是损失协调机制的直接证据，不优先扩张其他骨架。

D_no_routing 只能单独验证路由的增量，不能把它解释为“所有局部状态设计”的完整消融；论文需要单独主张状态损失贡献时，再登记 D_base_loss，不默认本轮扩成大矩阵。

A/B/C/D 相同 backbone trainability、额外 update 数、数据清单、采样、增强、base loss、优化器/LR schedule、seed、模型选择和推理协议。prior 预训练成本/额外监督单列，而不是声称整体计算量完全相等。

native-start 主比较可用 matched joint_finetune；先做 adapter-only smoke/机制 pilot 亦可，但四组不能混用冻结/解冻模式。adapter-only 下 A 没有新增可训练参数，只能作为固定 N0 reference，不称匹配继续训练；正式因果表以统一 joint_finetune 为优先方案。

参数量对照可按已有 R329 generic adapter 设计增补，先完成主四组和协调对照；旧 generic 结果不自动适用于新模块。跨数据集迁移若用于核心，只要来源和对照透明即可，不擅自用“全部必须从零 native”扩张成本。

## 9. 指标与验收：区域守底线，结构看收益

### 9.1 区域约束

两个数据集完整报告 Dice/DSC、IoU；“只有零点几个百分点提升”完全可作为区域收益，不设必须大涨的门槛。

首轮沿用现有严格点估计 gate：相对 N0，Dice 和 IoU 不低于基线；对 A 另行执行并报告同一 gate。N0-relative 总收益和 A-relative 先验增量分别判定；只超过 N0 不能宣称已证明相对同预算无先验模型的优势。实现误差容差与可接受性能损失不是一回事，不能借浮点容差接受实质下降。

需采用非劣效容差时必须事前单列，并区分点估计与 CI 结论；不能根据已见结果临时放宽。当前模板 margin=0，是保守继承，不是声称用户指定了统计阈值。

### 9.2 TSRS 主要结构评价

保留 R201：gap_region_fp_rate、component_merge_rate；同时给出 Boundary F1/IoU、Surface Dice、Recall、component_count_mae/delta、HD95/ASSD。说明 gapFP 背景带与 component proxy 的范围，不能把它们说成精确解剖骨缝测量。

所有模型共用冻结的 GT/ROI。若补充已有固定 seam ROI，报告它是否来自先验候选，避免把 prior-conditioned ROI 说成完全独立的人工真值。

### 9.3 RAM 主要结构评价

保留 R329：overall member、overlap-region、overlap-pair 的 DSC/NSD/VOE/MSD/RAVD；重点查看 overlap_msd_px、pair_msd_px、overlap/pair DSC/NSD 和 msd_fail_rate。

MSD 平均值降低但失败率上升不得判“结构改善”。固定 train top-pair list，不从 val/test 挑 pair；GT 空 pair 的排除和有效样本数必须报告。RAM seam 固定 ROI FP 作为次要保护/诊断，不要求与 overlap 获得同等幅度提升。

### 9.4 数值目标与成功层级

R351 已登记的 **5% 相对结构改善**可继承作首轮探索目标（TSRS gapFP/merge；RAM overlap/pair MSD），明确这是历史探索门槛，不是用户给定值、统计显著性或临床阈值。baseline=0 时不得除零，应报告绝对变化及不可计算的相对变化。

分开报告：工程通过；相对 N0 的结构收益；相对 A 的增量；D 相对 B/C 是否保持主要单先验收益；D 相对 D_no_routing 的协调贡献。D 没胜过最佳单先验不能藏，但不是更换论文主题的理由。

次要结构可以收益较小；不能用明显破坏另一类结构来换主结构改善。对 TSRS 无真实 overlap GT 的部分标 N/A/未验证，不填 0 分冒充“已无重叠错误”。

### 9.5 模型选择与统计

先冻结 validation checkpoint selection：区域 gate 优先，其后依据预登记主要结构分数排序；若均未过 gate，保存最优失败候选但不标 PASS。单一 checkpoint 绑定全部指标，不拼 epoch。

pilot 一个配对 seed；正式需要稳健结论时同一组至少 3 seeds 覆盖主要对照。逐 case 配对统计，存在 patient/group 时按组 bootstrap；不把 14 个骨成员当独立病人。当前不运行 test，不承诺历史 test 从未被看过。

## 10. 数据与变换检查

TSRS 原始 Epiphysis 与 Articular-Surface 不混用；original-val96、历史 train875/本地874 的 `1833.png` 差异逐 ID 核查。RAM train425/val69 作为历史预期，若新环境不同，先解释再锁定，不强行删文件凑数。

image、GT、S、O、valid、observation masks 必须同步空间变换。RAM `_R` canonical flip 一次；不得二次翻转 R323 prior。首轮延续无几何增强受控协议可以，但 A/B/C/D 均一致。

TSRS 保留原 nnU-Net 滑窗/TTA/回采样；冻结先验应在明确的全图坐标生成后同步裁剪，不因每个 patch 的 min-max 产生不同阈值含义。overlap 概率不得重新 min-max。标签读取与 baseline evaluator 不一致时硬停并记录，不能单改新算法 GT。

新 evaluator wrapper 在旧函数前核验完整 case IDs、尺寸、阈值、valid 与 ROI；缺失预测或坐标不符报错，禁止依赖旧代码的 skip/resize 容错。不得修改旧历史 evaluator 和结果；确有 bug 时创建版本化修复，所有对照重评并列旧新结果。

## 11. 代码落点：先复用，再最小隔离新增

下表“新增”为拟实现，不代表本交付包已经实现训练代码。

| 现有文件 | 处理要求 |
|---|---|
| `scripts/r351_dual_prior.py` | 保留旧模块；新模块显式 O 输入、S/O 信息隔离；不简单重命名了事 |
| `scripts/nnunet_trainers/nnUNetTrainerR351DualPrior.py` | 不直接作为新 native-start trainer；移除成熟 Dataset204/冻结/RAM refiner 强制耦合，新增版本化入口 |
| `scripts/train_r351_ram_dual_prior.py` | 复用 loader/评估/gate 思路；新增 native-start、多 arm、统一 trainability 入口 |
| `scripts/train_r351_overlap_prior.py` | 复用真实 RAM target；按需新增 image-only 变体及 TSRS 弱迁移 provenance |
| `scripts/r351_data.py` | 保留 canonical/valid 契约，统一变换字段 |
| `scripts/r350_unified_interaction_loss.py` | 复用基础项；补局部可观测性、weak 类型、valid-aware mass |
| `scripts/r349_parameter_gradient_gate.py` | 修复 None 投影，清晰 cap，增加回归；不覆盖旧结果 |
| `scripts/run_r201_unified_eval.py` | 只复用冻结纯指标函数，不运行默认 test main |
| `scripts/evaluate_r201_single_split.py` | 新 wrapper 先做 manifest/坐标强检查 |
| `scripts/evaluate_r329_ram_official_metrics_val.py` | 保留 R329 定义、失败率、pair 列表语义 |

建议隔离新增：

```text
scripts/dual_prior_v3/contracts.py
scripts/dual_prior_v3/priors.py
scripts/dual_prior_v3/local_supervision.py
scripts/dual_prior_v3/loss_and_routing.py
scripts/dual_prior_v3/run.py
scripts/dual_prior_v3/evaluate.py
scripts/nnunet_trainers/nnUNetTrainerDualPriorV3.py
scripts/tests/test_dual_prior_v3_*.py
run_dual_prior_v3_pilot.sh
```

如已有等价入口，优先复用；不另造与历史框架不兼容的大系统。run 的 stage/arm 必须显式，默认 preflight，无 test stage、无跨骨架 stage、无自动长训。

## 12. 执行阶段与产物

### M0 · 资产/基线与主线锁定

输出 `MAINLINE_LOCK.md`、`ASSET_LOCK.json`、`BASELINE_REPRODUCTION.json`。查原生权重、来源、无先验状态、数据清单、evaluator 和输出契约；完整验证集重现 N0。缺权重标 BLOCKED_ASSET，不从头重训顶替。

### M1 · 先验语义/标签/局部监督

输出两个数据集各自的 `PRIOR_SEMANTICS.json`、`LOCAL_SUPERVISION_AUDIT.json`。四个先验各有状态，TSRS overlap unknown/weak 不隐藏。统计同图 seam+overlap-support 共现、每类 mass、未知占比、可信度与 prior conflict；明确哪些是 GT 统计、哪些仅为 weak prediction。

先实现本地可验证监督及弱证据契约，不因 TSRS 缺真值而停止 RAM 的真实 overlap 审计，也不因有 RAM 真值就宣称 TSRS 已验证。

### M2 · 接口与协调修复

运行 `ACCEPTANCE_TESTS.md` 所列合成/契约测试；修复 overlap 独立通路、R349 None 缺陷、valid 和 arm 信息隔离。输出 `CODE_CHANGE_MAP.md`、`UNIT_TEST_RESULTS.json`。

### M3 · 两数据集真实 batch 与可视化

每数据集至少 1 个真实 batch；零初始化身份、有限 loss、关键末层梯度/参数更新、through-backbone 输入梯度、无 GT 推理、checkpoint 保存重载和 shape 契约均通过。零残差初期不要求所有隐藏层梯度立刻非零。

每数据集优先 12–20 个训练病例，固定选择规则覆盖 seam、多 overlap/weak-overlap、两类同图、空状态、困难边界。显示 X、GT、S、O、native/dual 预测、I_seam/I_overlap/unknown、误差图。不得手工只挑好例，不能把 weak 图标成 overlap GT。

输出 `REAL_BATCH_SMOKE_<dataset>.json`、`QUALITATIVE_INDEX.json`。这不是泛化效果验证。

### M4 · 有界 native-start pilot

M0–M3 满足后，两个数据集运行配对 A/B/C/D。默认只拟定 2 epoch 级别的固定小预算；TSRS 可沿用每 epoch100 updates 的量级，RAM 可按固定全 train pass 定义，具体在启动前按真实 sampler 锁定。不同数据集可以不同预算，同一数据集四组不能不同。

只用 train/original-val；明确 GPU、剩余空间、额外训练许可和隔离目录后执行。完成后补 D_no_routing；负结果如实记录。不得因有任意单点改善就自动开始 30 轮。

输出 `PILOT_MATRIX.csv/json`、单 checkpoint 全指标、paired deltas、routing 日志、资源消耗及失败项。

### M5 · 本轮收束

输出 `HANDOFF_STATUS.md`：实际命令、修改 diff、每个 stage 的 PASS/FAIL/NOT_RUN、资产/监督缺口、是否有验证集证据支持继续。正式长跑需冻结独立协议；跨骨架和最终 test 不属于当前执行范围。

## 13. 硬停与科研负结果区分

硬停当前受影响阶段：baseline 无法复现；标签/坐标错误；GT 进入推理；S/O 未隔离却进入 clean 主表；未知 overlap 被写成真值；aux routing 反例未修复；test 用于设计；缺预测被忽略；预算/起点不一致；为涨点改 evaluator。

不等于项目报废：TSRS weak prior 没有增益、D 不胜 B/C、结构改善不足5%、个别 seed 退化。记录原因和主张边界，可在既定研究主线内作有界修复，不能擅自转成 OOF completion 论文。

最终汇报必须同时回答：四个先验是什么、各自证据来源是什么、局部 loss 真正在哪生效、协调是否缓解冲突、相对 N0/A/B/C 有哪些收益与代价。不能只写“所有测试通过”或只给一张 Dice 表。
