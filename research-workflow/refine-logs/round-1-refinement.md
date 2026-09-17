# Round 1 Refinement

## Problem Anchor

- Bottom-line problem: 建立一个与基础分割网络解耦的儿童手部解剖先验，使同一先验在 nnU-Net、ARAA、YOLO+SAM、DINOv3/U-Net 等不同模型上都能改善骨缝粘连、边界质量和解剖一致性。
- Must-solve bottleneck: 儿童骨骺随年龄、性别和个体成熟度高度异步变化，固定模板会把正常发育差异误判为分割错误。
- Non-goals: 不设计新 backbone；不预测完整 foreground atlas；不使用 clean-test-v2 调先验；不鼓励骨头前景连通；不假设实例 ID 固定。
- Constraints: 仅 TSRS_RSNA-Epiphysis；R201；原始数据不变；先验和集成规则跨模型冻结；必须先回连并审计 RSNA metadata。
- Success condition: 同一冻结先验在至少 4/5 个异构模型上改善骨缝、边界和组件指标，同时 Dice/IoU/Recall 非劣。

## Anchor Check

- Original bottleneck: 相邻骨粘连和骨缝负空间被模型填满，同时儿童正常发育变化很大。
- Why the revised method still addresses it: 先验只描述“哪些中心应存在、哪些存在的相邻中心必须被背景分开”，不再拉预测去匹配平均前景形状。
- Reviewer suggestions rejected as drift: 不引入全分辨率 diffusion、第二个 correction network 或模型专属 adapter。

## Simplicity Check

- Dominant contribution: Development-Conditioned Negative-Space Graph Prior (DC-NSGP)。
- Components removed: dense foreground occupancy atlas、通用 set NLL、可交换 prototype slots、dense image-to-prior decoder。
- Remaining mechanism: global maturity latent + presence-conditioned anatomy graph + negative-space connectivity energy。
- Why smallest adequate: 三者分别解决异步成熟、中心数量可变和骨缝粘连；再删任何一项都会失去问题必要条件。

## Revised Proposal

### Method Thesis

儿童骨骺拓扑不是固定的，但在给定发育状态下，特定骨化中心存在以及特定相邻结构应保持分离的概率是可建模的；一个冻结的 presence-conditioned negative-space graph prior 可以在不规定平均手形的情况下正则化任意分割模型。

### 1. Metadata and developmental state

定义 chronological age (a_c)、sex (s) 和潜在发育状态 (z_{dev})。RSNA radiographic bone age (a_r) 若是骨龄评估标签，则禁止在测试时输入。

```
q(z_dev, z_pose | global_pool(g(x)), a_c, s)
```

- g 是冻结的全片 encoder；
- global pooling 后才进入 latent head；
- 无 spatial skip、无 dense feature map 传给先验；
- radiographic bone age 只可作为训练期 ordinal teacher：
  `L_ord = loss(h(z_dev), a_r)`，部署时删除 h；
- 若 chronological age 缺失，则用 X 光+sex 推断并对年龄条件边缘化；
- sex 只是弱协变量，不定义确定性男女模板。

### 2. Hand-centric canonical coordinates

统一左右手、旋转、hand ROI、全局长度和掌宽。先验使用相对坐标：

- digit-ray 方向；
- epiphysis center 间沿 ray 的相对距离；
- center area / hand area；
- center spacing / hand length；
- corridor width / local bone scale。

只移除拍摄姿态和全局尺度，不归一化掉指骨相对长度差异。若当前标签只有骨骺、没有完整指骨 shaft，则论文只能声称建模“骨骺中心间距/相对尺度”，不能把它写成直接测量指骨长度；实际指骨长度需要额外 landmark 或 shaft annotation。

### 3. Partially identified developmental graph

不用任意 slots，而定义分层图：

```
G(z) = {V(z), E(z)}
v_i = (family/ray, presence p_i, center μ_i, covariance Σ_i, scale ρ_i)
```

节点身份采用稳定的解剖 family/ray，例如 distal radius/ulna、五条 metacarpal/digit rays、proximal/middle/distal phalangeal levels。现有实例 masks 通过 hand-centric optimal transport 匹配到 family/ray；允许 unmatched 和 absent。

presence 使用随 z_dev 的 monotone spline 或 ordered hazard：

```
p_i(z_dev, s) = sigmoid(monotone_spline_i(z_dev) + sex_bias_i)
```

它区分：

- developmentally absent；
- present but radiographically weak；
- present but missed/merged by segmentation。

若标签无法可靠区分前两者，prior 以 calibrated uncertainty 合并，不做强制判断。

边只在两节点同时高概率存在时激活：

```
c_ij = p_i p_j P(separate_ij | z_dev, geometry)
```

因此低龄儿童缺少某中心时不会被强制创造虚假骨缝。

### 4. Negative-space corridor distribution

对每条激活边，低秩 anatomy basis 根据 z_dev、z_pose 生成两节点位置/尺度分布，并由 pairwise signed-distance geometry 推导背景 corridor：

```
C_ij ~ p(corridor | v_i, v_j, z_dev)
```

先验只生成 corridor mean/uncertainty，不生成完整 foreground mask。跨成熟 modes 均稳定的 corridor 置信度高；变化大的位置方差高。

### 5. Model-independent separation energy

任意模型输出经统一接口转换为 soft foreground probability p。对每条高置信 edge，计算两个节点支持区之间是否存在穿越 corridor 的 soft foreground connection：

```
E_sep(p) = [Σ c_ij SoftConnect(p; A_i, A_j, C_ij)]
           / [ε + Σ c_ij]
```

SoftConnect 可用 differentiable soft-min path/min-cut 或背景通道的 Euler violation approximation。它只惩罚本应分开的结构被前景桥接，不惩罚正常 foreground shape variation。

置信规则：

- c_ij 低时不激活；
- corridor uncertainty 高时权重下降；
- early training warm-up 不启用；
- prior gradient 仅作用于 corridor 和局部 connection path；
- 不允许自由全图 add/delete。

### 6. Cross-backbone normalized integration

不同模型输出统一为 normalized logits/soft mask。使用固定 EMA scale normalization：

```
L = L_native / EMA(|L_native|)
  + λ E_sep / EMA(|E_sep|)
```

所有 backbone 使用同一 prior、λ、EMA decay、warm-up 和 edge confidence threshold。论文主张的是固定 prior 和固定归一化规则，而不是未经校准的相同绝对梯度。

### 7. Training sequence

1. Metadata join：用 case ID 回连原始 RSNA CSV，区分 chronological age 与 radiographic boneage，报告 coverage 与 split 分布。
2. Graph identity audit：验证 family/ray assignment 的稳定性；无法稳定识别的节点合并成 family-level node。
3. Prior pretraining：仅 original train；学习 q(z)、presence curves、geometry covariance 和 corridor uncertainty。
4. Calibration：original val 只校准一次 graph coverage、corridor coverage 和全局 λ/threshold，不做 per-model tuning。
5. Backbone training：冻结 prior，接入多个模型的 output probability。
6. 协议冻结后 clean-test-v2 一次性 final evaluation。

### 8. Core validation

- Models: nnU-Net 2D、ARAA/DANet、YOLO+SAM R110、DINOv3 R130、high-res U-Net。
- Baselines: no prior；fixed average seam atlas；age/sex hard-bin graph；DC-NSGP。
- Primary metrics: merge rate、component count MAE、gap FP、Boundary IoU/F1、Surface Dice、HD95、ASSD。
- Non-inferiority: Dice/IoU/Recall。
- Main evidence: same prior and normalized integration improves at least 4/5 models。
- Essential ablation: metadata-only z vs X-ray latent z；无 presence conditioning；无 uncertainty；无 EMA normalization。

### 9. Main risks

- Metadata only provides radiographic boneage: use it only as teacher; deployed z uses X-ray+sex.
- Family/ray identity cannot be recovered from epiphysis masks: reduce graph granularity rather than fabricate anatomical IDs.
- z encoder learns segmentation: enforce global pooling, low-dimensional latent, no spatial decoder, and contrast/augmentation invariance.
- Prior harms young/rare cases: require conditional coverage calibration and abstention for low-confidence edges.
- Different backbones react differently: use normalized interface and report per-model gradient/prior activation statistics without tuning individual λ。

