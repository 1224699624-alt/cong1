# Research Proposal: Development-Conditioned Probabilistic Anatomy Prior for Pediatric Hand Epiphysis Segmentation

## Problem Anchor

- Bottom-line problem: 建立一个与基础分割网络解耦的儿童手部解剖先验，使同一先验在 nnU-Net、ARAA、YOLO+SAM、DINOv3/U-Net 等不同模型上都能改善骨缝粘连、边界质量和解剖一致性。
- Must-solve bottleneck: 儿童骨骺随年龄、性别和个体成熟度发生高度异步变化；骨化中心数量、显隐状态、指骨长度、骨块间距和骨缝宽度均非固定。确定性平均模板会把正常发育差异当成错误。
- Non-goals: 不设计新的主分割 backbone；不通过 clean-test-v2 调年龄分箱、阈值或先验权重；不鼓励骨头前景整体连通；不假设原始实例 ID 或简单左右排序稳定。
- Constraints: 仅 TSRS_RSNA-Epiphysis；R201 统一评估；原始数据结构不变；clean-test-v2 仅最终冻结评估；先验应能以相同参数作用于多个模型。当前本地 split 未附年龄/性别表，必须由 RSNA case ID 回连原始 metadata 并审计覆盖率。
- Success condition: 冻结同一先验和同一权重，在至少 4/5 个异构模型上改善 Boundary IoU/F1、Surface Dice、HD95/ASSD、gap-region FP、component merge rate 和 component count MAE，同时 Dice/IoU/Recall 满足预定义非劣界。

## Technical Gap

2024–2025 的研究已经证明三条路线有效：TMI 2024 Explicit Shape Priors 可跨 CNN/Transformer 注入全局与局部形状先验；JBHI 2024 的 SSM adaptive fusion 在多种 CNN backbone 上融合统计解剖知识；TMI 2025 Fast χ 使用与任意分割网络兼容的 topology violation map 驱动统一修正。TEDS-Net 和 Deep Closing 进一步说明形变模板和学习型形态先验可产生解剖可信输出。

但这些工作多假设结构拓扑相对固定，或鼓励管状前景连通。儿童手片的关键恰好相反：成熟过程是多模态且异步的，目标是保持相邻骨之间的背景负空间。仅按年龄/性别建立固定 atlas 不足；仅从图像预测一个 dense prior 又容易退化为第二个分割网络。

最小缺失机制是：一个低容量、概率化、发育条件化的先验分布，它只描述低频解剖布局、骨化中心存在性和高置信骨缝负空间，并将个体差异作为不确定性带，而不是强制平均形状。

## Method Thesis

- One-sentence thesis: 用年龄/性别作为群体成熟先验、用原始 X 光低频形态推断个体成熟 latent，生成多模态概率解剖场，并以置信区间外惩罚将其作为冻结、模型无关的训练正则。
- Why this is the smallest adequate intervention: 只新增一个离线先验生成器；基础模型、decoder 和推理结构均不改变。
- Why this route is timely: 它结合了显式 shape prior、SSM 多模态统计和 arbitrary-network topology violation 思想，但针对儿童发育异质性将确定性模板升级为条件概率分布。

## Contribution Focus

- Dominant contribution: Development-Conditioned Probabilistic Anatomy Prior (DCPAP)，一种对年龄、性别、个体成熟度和可变骨化中心集合进行联合建模的模型无关先验。
- Optional supporting contribution: 背景骨缝优先的 prior energy，将“前景连通”反转为“相邻骨负空间保持”。
- Explicit non-contributions: 不提出新 backbone；不把年龄预测、性别分类或 diffusion generation 作为并列贡献；不宣称 fixed topology 适用于所有年龄。

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: 任意已有分割网络；可选冻结的通用 X 光图像 encoder 仅提取低频成熟特征。
- New trainable components: 一个低容量条件 prior generator；无第二个 correction network。
- Tempting additions intentionally not used: 不使用全分辨率 diffusion mask generator，不用独立 landmark network，不做模型专属 adapter，不为每个 backbone 单独调先验。

### System Overview

```text
RSNA metadata: chronological age / sex (if available)
                           \
raw X-ray -> pose normalization -> low-frequency maturity encoder -> q(z|x,a,s)
                                                           |
train instance masks -> canonical set matching -> conditional prior bank
                                                           |
                              occupancy μ/σ + seam μ/σ + presence/graph distribution
                                                           |
any backbone probability map ------------------------------+
                                                           |
                 confidence-bounded prior energy during training
                                                           |
                         segmentation prediction
```

### Core Mechanism

#### 1. Metadata and leakage policy

必须区分 chronological age 与 RSNA radiographic bone-age label。如果 segmentation 是骨龄评估的上游，ground-truth boneage 不能作为推理输入。推荐协议：

- chronological age 和 sex 若临床可用，可作为条件输入；
- radiographic boneage 只可作为 prior teacher，不能进入下游模型推理；
- 若只有 boneage label，则训练 q(z|x,s) 去蒸馏成熟度，测试只用 X 光和 sex；
- sex 缺失时对 sex 条件边缘化，不丢弃病例。

#### 2. Pose/scale canonicalization

先在 hand ROI 中统一左右手方向、旋转、全局长度和掌宽。形状使用相对坐标和长度比，而非绝对像素。注册仅消除拍摄姿态/尺度，不抹去指骨相对长度和骨化中心大小，因为后者正是成熟度信号。

#### 3. Permutation-invariant anatomical set

项目历史表明组件数约 2–29、实例 ID 与简单空间顺序不稳定。因此把实例表示为集合：

```
B = {(presence_i, centroid_i, log-area_i, length_i/hand_length,
      orientation_i, signed-distance patch_i)}
```

训练时使用 Hungarian/optimal-transport matching 对齐到可学习 prototype slots；slot 允许 absent，不要求每张图拥有同样骨化中心数。相邻骨关系用稀疏图表示：相对位置、长度比、最小间距和背景通道宽度。

#### 4. Continuous, multimodal developmental conditioning

年龄不分硬 bin，而用连续 spline/Fourier embedding；sex 使用 embedding；图像只经过低分辨率/低频 encoder，输出成熟 latent z。条件先验为 mixture distribution：

```
p(P | x,a,s) = Σ_k π_k(x,a,s) p(P | z_k,a,s)
```

mixture 用来表达同龄同性别儿童的异步成熟，例如某些骨化中心已出现、另一些尚未出现。prior generator 只输出 64×64 或 128×128 的低频场，防止它成为第二个精细分割器。

#### 5. Three probabilistic prior outputs

- Foreground occupancy field: 每个位置属于合理骨结构的均值 μ_fg 与方差 σ_fg。
- Background seam field: 相邻骨之间应保持背景的均值 μ_seam 与方差 σ_seam。
- Set/graph distribution: 骨化中心 presence、组件数量分布、长度/面积比例、邻接间距分布。

关键不是只输出均值，而是输出 tolerance。青春发育差异大的位置具有高 σ，先验不强制修改；只有跨年龄/成熟原型都稳定的区域才施加强约束。

#### 6. Confidence-bounded prior energy

对任意 backbone 输出概率 p，不强迫其接近平均模板，而只惩罚落在条件可信区间外的预测：

```
L_prior = λ_fg Σ w_fg [|p-μ_fg|-κσ_fg]_+²
        + λ_seam Σ w_seam [p-(1-μ_seam)-κσ_seam]_+²
        + λ_set NLL(Q(p); presence/graph distribution)
```

其中 Q(p) 是输出的低频集合/图统计。高不确定区域权重低；高置信骨缝区域抑制错误前景；对少见但合理的骨化中心不做硬删除。

为避免损伤 Recall，加入 one-sided trust rule：prior 主要在模型低置信区域工作；模型高置信前景只有在 prior 跨 mixture components 都判为高置信骨缝时才受罚。

### Integration into Any Segmentation Model

1. 独立训练 prior generator，仅用 original train metadata、X 光和实例 mask。
2. 冻结 prior generator。
3. 对每个基础模型保留原有损失，加同一个 `L_prior`、同一个 λ 和同一 curriculum。
4. 先完成 backbone 原损失 warm-up，再逐步开启 prior；防止早期错误预测被 prior 放大。
5. prior 不读取 backbone feature，只读取 raw X-ray/metadata 并约束最终概率图，因此 CNN、Transformer、SAM-derived pipeline 接口一致。

### Training Plan

- Stage A: metadata join 与 leakage audit；覆盖率、年龄/性别分布、split 平衡。
- Stage B: canonicalization 与 set matching audit；验证 prototype matching 对年龄连续变化稳定。
- Stage C: prior pretraining。目标为低分辨率 occupancy/seam distributions、presence 和 graph NLL；用 metadata dropout 训练缺失条件鲁棒性。
- Stage D: 冻结 prior，在多个 backbone 上用相同权重训练；original-val 只选择一次全局 λ/κ，不做 per-model tuning。
- Stage E: 协议冻结后统一 clean-test-v2 final evaluation。

### Failure Modes and Diagnostics

- Boneage target leakage: 若 boneage 是最终预测标签，禁止测试时输入；改用 image-inferred z。
- Prior collapses to age average: 检查 conditional coverage/calibration，要求真实 mask 落入预测区间的比例达到目标。
- Image encoder becomes a second segmenter: 限制输入分辨率、latent 维度和输出频率；做 metadata-only 与 image-conditioned ablation。
- Prototype identity instability: 使用 permutation-invariant matching，报告 slot presence entropy 和跨增强一致性。
- Minority maturity modes erased: 使用 mixture prior 与 variance-aware hinge，不用均值 Dice loss。
- Prior helps weak models but hurts strong models: 采用 uncertainty/trust gating，并坚持同一参数跨模型评估。

### Novelty and Elegance Argument

与 Explicit Shape Priors 的差别是先验不是单一确定性形状特征，而是面向发育异步性的条件分布；与 SSM adaptive fusion 的差别是无需为每个 CNN 构造候选融合器；与 TEDS-Net 的差别是基础模型和拓扑不固定；与 Fast χ 的差别是 violation 不只来自固定 Euler genus，而来自年龄/性别/成熟条件下的骨化中心 presence、组件关系和背景骨缝概率。

论文的唯一中心问题是：如何让发育变化很大的儿童解剖先验既有约束力，又不把正常个体差异抹平。

## Claim-Driven Validation Sketch

### Claim 1: 条件概率先验可跨模型稳定提升解剖一致性

- Minimal experiment: 冻结同一 prior、λ、κ，应用于 nnU-Net、ARAA/DANet、YOLO+SAM R110、DINOv3 R130、high-res U-Net。
- Baselines: 原模型；固定平均 atlas；age-sex hard-bin atlas；DCPAP。
- Metrics: Boundary IoU/F1、Surface Dice 2/5、HD95、ASSD、gap FP、merge rate、component count MAE；Dice/IoU/Recall 非劣。
- Expected evidence: 至少 4/5 模型结构指标改善，且强模型不被系统性拖低。

### Claim 2: 图像成熟 latent 对同龄异步发育是必要的

- Minimal experiment: metadata-only vs image-only vs age/sex+image mixture prior。
- Metric: prior calibration/coverage、presence NLL、跨成熟亚组的分割收益。
- Expected evidence: 条件 mixture 在同年龄组内的不同骨化中心数量和长度比例上显著优于 hard bins。

## Experiment Handoff Inputs

- Must-prove claims: 跨模型一致增益；概率先验不抹平正常发育差异。
- Must-run ablations: deterministic mean atlas；age/sex hard bins；无 image maturity latent；无 variance tolerance；无 seam term。
- Critical data: RSNA metadata join；实例 mask canonicalization；年龄/性别/成熟亚组 stratification。
- Highest-risk assumptions: metadata 是否完整；当前实例标签是否足以建立稳定 prototype set；同一 prior weight 能否同时适配弱/强模型。

## Compute & Timeline Estimate

- Prior pretraining: 单 GPU 约 8–20 小时，取决于低频 encoder。
- Backbone experiments: 先验本身冻结；每个模型仅增加低分辨率 prior forward 与 loss，额外开销预计较小。
- Data/annotation cost: 不新增像素标注，但需要 metadata 回连和可能的 anatomical slot audit。
- Timeline: 先完成 2–3 天 metadata/set feasibility，确认先验 calibration 后再决定是否进入跨模型训练。

