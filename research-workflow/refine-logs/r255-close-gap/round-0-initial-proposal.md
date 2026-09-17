# Research Proposal: Development-Conditioned Close-Gap Confidence Suppression

## Problem Anchor

- **Bottom-line problem**: 儿童手部 X 光骨骺分割中，主要粘连发生在腕骨/掌骨近端簇。低骨龄儿童骨化中心少、骨间距大，通常不需要扩大前景或额外干预；高骨龄儿童骨化中心逐渐发育、尺寸增大、局部骨缝缩窄，尤其是骨边界局部逼近时容易被分割模型连成同一前景组件。
- **Must-solve bottleneck**: 在不切掉真实骨头、不凭空生成骨缝标签、不让低骨龄病例产生额外骨化中心的前提下，降低高骨龄近距离骨对最窄瓶颈处的前景置信，阻断假连接。
- **Non-goals**: 不追求靠整体前景缩小提升 Precision；不鼓励前景骨头连通；不把 component count 接近当成骨对正确；不以 Dice/IoU 超过 nnU-Net 为唯一目标。
- **Constraints**: 只做 TSRS_RSNA-Epiphysis；原数据结构不修改；只使用 bone mask 的合法 y=0 背景；无 y=0 时跳过；clean-test-v2 仅最终一次评估；首轮仍以 nnU-Net 2D 为骨架。
- **Success condition**: original-val 的 1–4 px close-gap FP、same-component merge rate 明显下降，Precision 与 Boundary IoU/F1 不下降，Recall/Dice 无明显退化；定性图中高骨龄腕骨簇的最窄连接被断开，而低骨龄病例不出现前景扩大或新骨化中心。

## Technical Gap

R254 的 selected-negative confidence 从 0.839 降到 0.051，但 gap-region FP 增加 0.041。原因不是 loss 没有生效，而是 anchor 定义错位：形态距离门槛排除了 1–2 px 最近骨缝，generic adjacent-pair prior 又没有区分发育阶段、骨化中心是否存在及腕骨局部骨对。与此同时 separately-normalized core BCE 造成前景膨胀，表现为 Recall 上升、Precision/Boundary/gap FP 下降。

年龄不能作为统一的“切割强度”。低骨龄时骨化中心少且间距大，prior 应 abstain；高骨龄时只有在两个骨化中心确实存在、当前预测把它们并入同一组件、局部 X 光仍显示分离证据时，才提高瓶颈背景抑制。

## Method Thesis

- **One-sentence thesis**: 用连续骨龄/性别条件的骨化中心存在性与骨对距离分布，只给“高发育阶段、同一预测组件、局部边界逼近且 X 光仍有分离证据”的最窄合法背景瓶颈分配负类置信抑制。
- **Smallest adequate intervention**: 保留 nnU-Net 原网络、原生 Dice+CE 和标准推理，仅新增训练期实例 sidecar pairing 与一个无参数/低参数的 close-gap coefficient。
- **Dominant contribution**: development-conditioned merge-specific confidence allocation，而不是新的分割骨干或后处理切割器。

## Proposed Method

### Complexity Budget

- **Frozen/reused**: nnU-Net 2D backbone、预处理、增强、deep supervision、原生 Dice+CE、推理和阈值。
- **New trainable components**: 无必需新网络；首版 prior 为训练集条件统计。
- **Training-only sidecar**: 原实例 ID map 与 binary target 同步几何增强，只用于区分骨化中心身份。
- **Excluded**: dense seam decoder、固定平均手形、推理后 watershed/cut、前景 closing、额外 core BCE。

### Development-conditioned state

对腕骨/掌骨近端 ROI 的解剖槽位或稳定空间原型估计：

```text
pi_i(a,s) = P(ossification center i is present | bone age a, sex s)
D_ij(a,s) = conditional distribution of normalized local boundary distance
```

骨龄采用连续核平滑，并输出独立病例有效样本量 ESS。低支持年龄区间降低系数，不用硬年龄分箱。

如果真实部署无法获得骨龄，则训练时将骨龄作为 teacher，推理时对图像估计的发育表征边缘化；本数据集 pilot 可以先使用已提供 boneage CSV 验证机制。

### Merge-specific pair gate

两个实例支持 i、j 只有同时满足以下条件才激活：

```text
present_i × present_j is reliable
same_pred_component(i,j) = 1
normalized_boundary_distance <= tau_close(a,s,i,j)
prediction confidence on both supports is sufficient
pair belongs to wrist/carpal-metacarpal ROI
```

低骨龄中心不存在时不构造 pair；已经分开的预测组件不再干预。

### Close-gap bottleneck

从两个实例边界最近点构造局部窄 corridor，并与预测组件的细颈/高前景桥相交。候选像素仍必须满足原 binary target 的 y=0。

对于 1–2 px 窄缝，不再要求 EDT 远离前景。可靠度为：

```text
q_gap = instance-distinctness
      × local X-ray valley/double-edge evidence
      × augmentation stability
      × annotation support
```

如果两实例之间完全没有 y=0，则严格跳过。

### Prior coefficient

```text
k_ij = stopgrad(
    support_ESS(a,s)
  × presence_confidence_i,j(a,s)
  × pair-specific developmental plausibility
  × same-component merge risk
  × local closeness risk
  × X-ray separation evidence
)
```

年龄大不自动意味着强切割；年龄只改变中心存在性和该骨对距离的合理分布。最终决策仍需当前图像与预测共同支持。

### Loss

```text
L_base = canonical nnU-Net Dice + CE

L_gap_bg = normalized BCE(background)
           on legal close-gap bottleneck pixels

L_gap_rank = softplus(
    margin + mean(logit_gap)
           - min(mean(logit_core_i), mean(logit_core_j))
)

L_total = L_base + alpha_bg L_gap_bg + alpha_rank L_gap_rank
```

排序项让骨缝 logit 相对骨核心更低，避免对不同曝光/发育阶段使用同一绝对阈值。alpha 继续由最终 logits 梯度比限制，且禁止改变全局前景权重。

## Failure Modes and Diagnostics

- **低骨龄 hallucination**: 监控按骨龄分组的新增前景面积、预测组件增量；低骨龄 pair activation 应接近零。
- **近缝仍被漏选**: 按 1–2、3–4、5–8、>8 px 记录候选覆盖与 anchor p(foreground)。
- **过切真实骨头**: 监控 cut/anchor 区域 GT foreground fraction，必须严格为 0；Recall/core confidence 非劣。
- **年龄分布稀疏**: 报告 ESS，低 ESS 自动 abstain。
- **标签无缝**: 无合法 y=0 时跳过，不从图像伪造 hard target。

## Minimal Claim-Driven Validation

### Claim 1: Close-gap specificity

- **Experiment**: nnU-Net native vs same-component close-gap loss，单 seed sanity。
- **Metric**: 1–2 px、3–4 px gap FP；same-component merge rate；Precision/Recall guardrail。
- **Pass**: close-gap FP 与 merge rate下降，Precision不下降，Recall/Dice无明显损失。

### Claim 2: Development conditioning is necessary

- **Ablation**: uniform pair coefficient vs age/sex presence-distance coefficient。
- **Metric**: 按低/中/高骨龄分层的 harmful activation、gap FP、foreground area change。
- **Pass**: 低骨龄 activation 更少且不扩大前景；高骨龄近缝改善集中在真正风险骨对。

### Deletion check

- canonical base + pair 与 R254 robust base + pair 对比，验证删除 core BCE 是否消除前景膨胀。

## Compute and Scope

- 首轮仅 2-epoch sanity + 10–15 epoch pilot。
- 通过 original-val gate 前不运行 clean-test-v2、不扩展其他骨干。
- 不增加推理时网络或后处理。

