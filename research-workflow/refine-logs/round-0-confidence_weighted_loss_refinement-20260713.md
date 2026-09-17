# Research Proposal: Prior-Calibrated Reliability and Valley Loss (PCRV-Loss)

## Problem Anchor

- Bottom-line problem: 在不依赖高质量骨缝标签、不构造反向背景 seam GT 的前提下，利用发育先验调整训练置信和损失系数，使不同分割模型更容易区分相邻骨头。
- Must-solve bottleneck: 当前监督是骨头 mask，边界和局部组件存在标注噪声；直接反转 mask 会把骨头标注误差完整复制成错误骨缝监督，而普通 Dice/BCE 又会平等拟合错误边界和真实骨头内部。
- Non-goals: 不生成伪骨缝 mask；不自由删除前景；不把模型自身高置信当作标签必然正确；不增加第二分割器；不使用 clean-test-v2 调权重。
- Constraints: 仅 `TSRS_RSNA-Epiphysis`；R201；原始数据不变；同一先验与 loss 规则跨模型；original train/val 开发；clean-test-v2 最终一次评估。
- Success condition: 在 Dice/IoU/Recall 非劣的同时，跨多个模型降低 merge rate、gap-region FP 和 component-count MAE，并改善 Boundary IoU/F1、Surface Dice、HD95/ASSD。

## Technical Gap

如果标签中存在可靠的骨头内部和远背景，但局部边界/骨缝有误，均匀 Dice/BCE 会过拟合噪声；若直接构造背景 seam label，又会把不合格骨头边界当作精确负监督。需要的不是新的 seam target，而是两个层次的置信调节：像素级 label reliability 决定原骨头标签应被拟合多强，候选对级 developmental prior 决定相邻骨头之间已有的可靠背景证据应被强化多少。

## Method Thesis

保留原始骨头标签作为唯一 hard target，以 stop-gradient label reliability 调节 pixel loss，以发育先验的 pair-separation probability 调节可靠负锚点上的局部置信谷值 margin；没有可靠负锚点时自动跳过，不凭空制造骨缝。

## Contribution Focus

- Dominant contribution: PCRV-Loss，一个同时处理 noisy bone masks 和相邻骨头置信分离的 prior-calibrated loss。
- Supporting contribution: original-train OOF/augmentation agreement 提供模型无关的 label reliability，避免用当前模型自信度自我确认。
- Explicit non-contributions: 不提出 background topology label、post-hoc cut editor、完整实例识别器或新 backbone。

## Proposed Method

### 1. Three reliability zones from the existing bone label

仅从原始 bone mask `y` 构造置信区域，不把取反后的 mask 当骨缝 GT：

```text
C_fg = erosion(y, r_core)                 # 高可信骨头内部
C_bg = outside(dilation(y, r_far))        # 高可信远背景
U_bd = remaining pixels                   # 边界/近骨区域，不确定
```

若 mask 有实例 ID，只在单张图内用于生成 component core；不假设 ID 跨图稳定。若只有 binary mask，则使用 connected components/watershed core，但不赋命名身份。

### 2. Stop-gradient pixel label reliability

```text
q_x = clip(q_morph(x) * q_ann(x) * q_oof(x) * q_aug(x), q_min, 1)
q_x = stopgrad(q_x)
```

- `q_morph`: core/far-background 高，边界带低；
- `q_ann`: 若 original/reannotated 两版本均存在，一致像素高、分歧像素低；无第二标注时取 1；
- `q_oof`: patient-disjoint OOF ensemble/多个 source predictions 的稳定度，不使用当前样本的 in-fold prediction；
- `q_aug`: 小幅增强逆变换后的标签/teacher prediction 稳定度；
- `q_min>0`: 防止困难但正确的边界完全失去监督。

`q_oof/q_aug` 只衡量稳定性，不能因为模型与标签不一致就直接判定标签错误；当前训练模型不能通过提高自身置信来提高权重。

### 3. Reliability-weighted native segmentation loss

按前景/背景分别归一化，避免远背景数量压倒骨头：

```text
L_wBCE = - mean_fg[q_x y log p]
         - mean_bg[q_x beta_x (1-y) log(1-p)]

L_wDice = 1 - (2 sum q_x p y + eps)
                  /(sum q_x p + sum q_x y + eps)
```

`beta_x=1` 默认；只有后述可靠 inter-bone negative anchors 才提高。边界不确定区使用 bounded robust loss（例如 GCE）替代高曲率 BCE：

```text
L_unc = mean_{x in U_bd} q_x * (1 - p_y^gamma)/gamma
```

### 4. Prior only supplies pair coefficients, not seam pixels

沿用冻结 HA-PDSP relation posterior，但只输出：

```text
s_ij = P(two prediction-derived candidates should be distinct | image context, geometry)
c_ij = epistemic/candidate/stability confidence
k_ij = stopgrad(s_ij * c_ij)
```

prior 不输出 corridor mask 或 background label。候选必须来自当前基础模型/OOF soft prediction，不由 GT 在推理时生成。

### 5. Reliable negative anchors between candidate pairs

对候选对 `(i,j)`，只从“原 bone label 已经标为背景”的像素中选择负锚点：

```text
N_ij = {x:
        y(x)=0,
        x lies in the geometric between-region of i,j,
        q_x >= tau_q,
        OOF/EMA background consensus >= tau_bg}
```

between-region 只限定搜索范围，不把整个区域赋成背景。若 `N_ij` 为空，pair loss=0；因此标签完全漏掉骨缝时，本方法不会伪造 cut。

### 6. Confidence-valley ranking instead of reverse-label loss

令两端可靠 bone cores 的置信为：

```text
c_i = mean_{x in core_i} p(x)
c_j = mean_{x in core_j} p(x)
v_ij = SoftMaxPool_{x in N_ij} p(x)   # 最危险负锚点的前景置信
```

定义：

```text
L_valley = mean_{valid pairs}
           k_ij * relu(m + v_ij - min(c_i,c_j))
```

它要求“骨头核心置信高于两骨之间可靠背景锚点至少一个 margin”，本质是调整相对置信而非监督完整骨缝。等价地，可在 `N_ij` 上设置：

```text
beta_x = 1 + lambda_pair * aggregate_{ij:x in N_ij} k_ij
```

但最终实现只保留 ranking 或 local background reweighting 其中一个，避免重复惩罚；优先保留 `L_valley`，因为其尺度更可解释。

### 7. Uncertain boundary consistency

不确定区域不使用 hard pseudo-seam，只在 EMA teacher/OOF ensemble 高度一致时做 soft consistency：

```text
L_cons = mean_{x in U_bd}
         q_cons(x) * KL(p_student(x) || stopgrad(p_teacher(x)))
```

teacher/OOF 不一致时该像素只保留低权重 `L_unc`，不生成 hard target。

### 8. Final loss

```text
L_seg = L_wDice + L_wBCE + lambda_unc L_unc
L = L_seg + lambda_valley L_valley + lambda_cons L_cons
```

各辅助项在最终 logits 上使用固定 gradient-ratio cap，使同一规则跨 backbone，不按模型单独搜索绝对 loss 权重。先 warm-up `L_seg`，待候选稳定后再启用 `L_valley`。

## What the Loss Can and Cannot Fix

- 可以：降低噪声边界对梯度的支配；保护高可信骨头 core；强化标签中已经存在的少量可靠 inter-bone background；使“两个骨头核心高置信、中间负锚点低置信”形成相对 margin。
- 不能：在标签完全把两个骨头连成一个前景、且 OOF/teacher 也没有稳定背景证据时凭空恢复骨缝。此时必须 skip，而不是伪造监督。

## Claim-Driven Validation Sketch（当前不执行）

### Claim 1: Reliability weighting improves noisy-label robustness

- Compare: native Dice/BCE；morph-only weights；morph+OOF reliability；full PCRV-Loss。
- Report: Dice/Recall、Boundary metrics、per-label-disagreement subgroup、loss/gradient mass in core vs uncertain band。

### Claim 2: Pair coefficients reduce merges without seam GT

- Compare: no pair term；uniform pair term；prior-calibrated `L_valley`；reverse-seam loss diagnostic only。
- Report: merge rate、gap FP、component-count MAE、harmful pair activation、valid-pair coverage、Recall non-inferiority。

### Cross-model requirement

- Same reliability construction, frozen prior, margin definition and gradient-ratio rule across at least 5 model families.
- clean-test-v2 remains locked until all rules are frozen on original train/val.

