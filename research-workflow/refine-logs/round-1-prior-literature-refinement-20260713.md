# Round 1 Refinement: Heterochrony-Aware Prediction-Derived Separation Prior

## Problem Anchor

- Bottom-line problem: 建立一个与基础分割网络解耦的儿童手部骨骺先验，使同一冻结先验能在多种分割模型上改善骨缝粘连、相邻骨误连接、边界质量和解剖一致性。
- Must-solve bottleneck: 儿童骨骺不仅随年龄和性别变化，同一个儿童的不同区域也可能异步成熟；固定形状、固定组件数或单一平均模板会把正常差异当成错误。
- Non-goals: 不设计新 backbone；不生成完整 foreground atlas；不鼓励骨头前景连通；不依赖固定实例 ID；不使用 clean-test-v2 调参；不直接声称建模完整指骨长度。
- Constraints: 仅 `TSRS_RSNA-Epiphysis`；严格 R201；不改变原始数据结构；先验、候选算法和集成规则跨模型冻结；仅 original train/val 建模和选择。
- Success condition: 同一冻结先验在至少 4/5 个异构模型上改善 Boundary IoU/F1、Surface Dice、HD95/ASSD、gap FP、merge rate 和 component-count MAE，同时 Dice/IoU/Recall 非劣。

## Anchor Check

- 主线仍是背景骨缝 separation prior，不是骨龄预测或完整形状生成。
- 删除固定九区成熟 latent 后，局部异步发育由“候选位置触发的局部骨化证据”表达；它不能发现或补回节点，因此不构成定位器。
- 删除 masked graph reconstruction、component dropout、独立 classifier+density 双头以及训练反馈型 violation map，避免六篇论文机制拼装。

## Simplicity Check

- Dominant contribution: 一个统一的发育条件 class-conditional relation posterior，判断 prediction-derived candidate pair 是否代表两个应分开的骨骺。
- Supporting mechanism: 固定 max-min foreground bridge energy，将该 posterior 变成跨模型的负空间约束。
- New trainable components: 至多两个——低维 global development/local evidence encoder；统一 relation model。
- Frozen/non-trainable: candidate graph、max-min connectivity、epistemic support、gradient-ratio interface、audit-only violation map。

## Revised Proposal

### 1. One-sentence thesis

用整手 X 光估计全局发育状态，并仅在基础模型已经提出的候选位置读取局部骨化证据；一个冻结的 class-conditional relation model 判断候选对是否应分开，再通过 max-min 前景桥能量惩罚错误粘连，从而在不生成 shape atlas、不固定组件数的前提下跨模型注入儿童发育先验。

### 2. Deterministic coordinate normalization

只做确定性的左右手统一、hand ROI、旋转和长宽尺度归一化。节点坐标以 hand width/height 归一化。粗区域标签只作为坐标 feature，不定义固定区域 latent，也不声称是具体骨骺身份。

### 3. Global development and prediction-triggered local evidence

冻结或轻量微调的 X 光 encoder 输出 feature map `F(x)`，但无 decoder、skip、heatmap 或 dense prior output：

```text
z_g ~ q(GlobalPool(F(x)), age_if_truly_available, sex)
a_i = PatchPool(F(x), stopgrad(candidate_center_i), fixed_radius_i)
```

- `z_g` 是整手低维 developmental posterior。
- `a_i` 只在基础模型 soft mask 已产生的 candidate center 上读取固定 patch pooled feature，表示 local ossification evidence，不命名为区域骨龄或局部成熟年龄。
- `a_i` 不能产生新节点、移动中心或输出空间 mask；没有候选就没有 local evidence。
- patch radius 由候选 basin scale 的冻结规则确定；所有 backbone 共用。
- chronological age 只在真实存在时输入；RSNA `boneage` 是 radiographic bone age，只能在 original-train 作可删除 teacher；sex 是 shrinkage weak covariate。
- 曝光、对比度、旋转、尺度增强前后约束 `z_g` 和同一候选 patch evidence 一致。

局部异步发育由同一儿童不同 candidate 的 `a_i` 差异表示，而不是由缺少标签的九个自由 latent 表示。

### 4. Shared prediction-derived graph

所有数据都走同一 stop-gradient pipeline：

```text
soft mask -> normalization -> distance transform -> local peaks
          -> watershed -> normalized nodes -> Delaunay -> frozen edge cap
```

节点仅含 center、area、basin scale、peak confidence、coarse coordinate 与 `a_i`。缺少中心时不创建虚拟节点；空图 abstain。当前 epiphysis label 只支持骨骺中心、面积、相对位置、间距和 gap，不支持 shaft length。

### 5. One class-conditional heteroscedastic relation model

将上一版独立 `p_sep` classifier、normality likelihood 和 masked reconstruction 合并为一个模型。定义不包含错误强度的静态 eligibility feature：

```text
e_static = [relative center displacement,
            log basin-area ratio,
            normalized basin scales,
            coarse coordinates,
            local evidence a_i,a_j]
```

明确排除：predicted gap width、foreground bridge strength、`Conn_ij` 和任何会因粘连本身变异常的 feature。

```text
p(e_static | y_sep, z_g) = N(mu_y(z_g), diag(sigma_y^2(z_g)))
P(y_sep | e_static,z_g) ∝ p(e_static|y_sep,z_g) P(y_sep|z_g)
```

- 默认 `K=1` heteroscedastic Gaussian；只有 original-val 的 held-out NLL/calibration 证明明显多模态时，才允许统一升级为小 `K` mixture，不能按 backbone 选择 K。
- `sigma_y` 表达 aleatoric developmental variation，决定关系约束应多强，但不能用于 OOD abstention。
- posterior 直接校准；没有第二个平行 classifier head。

### 6. Training distribution and provenance labels

先验训练使用两类 original-train 输入，全部通过同一候选 pipeline：

#### A. GT-derived structural corruptions

- bridge corruption：连接两个不同 GT components；候选 seed provenance 不同，`y_sep=1`。
- over-split corruption：分裂同一 GT component；provenance 相同，`y_sep=0`。
- 小幅 geometry/appearance jitter：只作鲁棒性增强。
- 背景、混合、无法唯一匹配 GT component 的 edge ignore；non-neighbor 不当负例。

删除 component dropout，因为当前方法不建 presence/count model；删除 masked attribute reconstruction，因为它不直接服务 separation posterior。

#### B. Original-train out-of-fold model predictions

仅靠合成 corruption 与真实 soft mask 存在域差，因此对 source backbone 生成 patient-disjoint OOF soft predictions：

- candidate selection 只看 OOF prediction，不看 GT；
- GT 只在候选产生后赋 provenance edge label；
- OOF logits/probabilities使用与部署相同 normalization 和 candidate algorithm；
- relation model 在 synthetic edges 与 OOF edges 上共同训练并校准。

```text
L_prior = class_conditional_NLL(e_static,y_sep,z_g)
        + lambda_cal * calibration_loss
        + optional lambda_ord * radiographic_age_teacher
        + lambda_inv * augmentation_consistency
```

### 7. Reliability-only activation gate

不再使用 raw relation likelihood 或 aleatoric variance 来回避“异常关系”。激活权重为：

```text
w_ij = P(y_sep=1 | e_static,z_g)
     * c_epi
     * c_candidate
     * c_stability
```

- `c_epi`: relation representation 到 original-train support 的 epistemic confidence，可由冻结 ensemble disagreement 或 calibration-set latent distance得到；它只看 `e_static`，不看 bridge/gap error strength。
- `c_candidate`: 两个独立、稳定 distance peaks/basins 的置信度。
- `c_stability`: 轻微输入/概率扰动下候选节点和 edge 是否一致。
- aleatoric `sigma_y` 用于 soft posterior，不作为 OOD veto。

因此严重 bridge 不会因为 gap/connection 异常而被 gate 丢弃；只有 candidate 本身不可靠或训练支持不足时 abstain。

### 8. Reversed-topology max-min bridge energy

对每条激活 edge，在固定 `64×64` crop 上计算：

```text
r_0 = stopgrad(source_seed)
r_{t+1} = max(r_0, min(p_crop, MaxPool3x3(r_t)))
T = 64
Conn_ij = mean(r_T[target_seed])

E_sep = sum(w_ij * Conn_ij) / (eps + sum(w_ij))
```

`Conn_ij` 是 source-target strongest path 的瓶颈前景概率。它只惩罚本应分开的候选被高置信前景路径桥接，不鼓励 foreground connectivity，也不会累计弱背景响应。

### 9. Cross-model output-space integration

```text
g_n = ||dL_native/dl||_2
g_p = ||dE_sep/dl||_2

if no_edge or g_p < tau_g:
    L = L_native
else:
    alpha = clip(rho * stopgrad(g_n/(g_p+eps)), alpha_min, alpha_max)
    L = L_native + alpha * E_sep
```

prior、candidate algorithm、calibration、gate、gradient ratio、warm-up 和阈值全部在 original train/val 冻结，跨模型共用。YOLO+SAM 只能接入可训练 mask decoder/refiner；完全冻结的黑盒输出不能通过 training loss 改善。

### 10. Audit-only violation map

```text
V = stopgrad(abs(dE_sep/dl))
```

`V` 只用于可视化最强桥路径、统计 intervention area 和错误分析；绝不乘回 loss、mask gradient 或参与同一次训练更新，因此不产生二阶/循环梯度。真正的梯度定位已经由 edge crop 和 max-min strongest path 完成。

## Exact literature inheritance

- Carneiro-Esteves：保留独立预训练、冻结 plug-and-play regularizer；反转其 foreground reconnect 目标。
- TEDS-Net：保留 topology/geometry 分离；删除固定 topology deformation atlas。
- Zhao et al.：保留 variance-aware statistical relation 与 selective activation；gate 只用 epistemic/candidate reliability，不用 normality likelihood。
- Deep Closing：保留 bridge/over-split corruption 和 topology-critical focus；删除 foreground closing 和 simple-point erosion。
- Explicit SPM：保留 global/local 职责分离；local 变成 candidate-triggered pooled evidence，而不是 dense prior map。
- Fast χ：保留 violation visualization；不设固定 Euler target、不增加 correction network、不让 violation map反馈训练。

## Claim-Driven Validation Sketch（本阶段不执行）

### Claim 1: 同一先验跨模型改善错误粘连

- 5 个异构模型使用同一冻结 prior/interface。
- 主指标：gap FP、merge rate、component-count MAE、Boundary IoU/F1、Surface Dice、HD95、ASSD；Dice/IoU/Recall 非劣。
- 必须报告每模型 candidate coverage、activation rate、abstention rate、harmful activation rate。

### Claim 2: 个体局部骨化证据比固定年龄模板更安全

- Ablations: no prior；age/sex hard bin；global-only；global+candidate local evidence；去 epistemic gate；synthetic-only vs synthetic+OOF。
- 分层：年龄/性别、组件数、低质量影像、局部证据不一致病例。

### Generalization test

- Leave-one-backbone-out：relation prior 不看某一 backbone 的 OOF 输出，再冻结测试其 candidate coverage、calibration 和结构收益。
- 该测试决定“跨模型先验”主张是否成立，不能只报告在所有 source backbone 混合训练后的结果。

## Protocol Lock

仅 original train/val 训练、校准和选择；clean-test-v2 在方法和所有阈值冻结后仅做一次 R201 final evaluation；不混入 Articular-Surface，不改变原始数据结构。本阶段不运行实验。

