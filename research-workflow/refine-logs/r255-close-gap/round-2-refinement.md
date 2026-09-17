# Round 2 Refinement: Local-Corridor Development-Conditioned Bridge Suppression

## Problem Anchor

- **Bottom-line problem**: 儿童手部 X 光骨骺分割中，主要粘连发生在腕骨/掌骨近端簇。低骨龄儿童骨化中心少、骨间距大，通常不需要扩大前景或额外干预；高骨龄儿童骨化中心逐渐发育、尺寸增大、局部骨缝缩窄，尤其是骨边界局部逼近时容易被分割模型连成同一前景组件。
- **Must-solve bottleneck**: 在不切掉真实骨头、不凭空生成骨缝标签、不让低骨龄病例发生错误碎裂或小骨化中心消失的前提下，降低高骨龄近距离骨对最窄瓶颈处的前景置信，阻断假连接。
- **Non-goals**: 不做全手 component-count 优化；不使用推理期骨龄；不通用收缩前景；不生成 seam mask；不增加推理模块。
- **Constraints**: TSRS_RSNA-Epiphysis only；原数据不改；anchor 必须 y=0；无可靠背景跳过；clean-test-v2 锁定；nnU-Net native base。
- **Success condition**: 1–4 px close-gap FP 和 false merge 下降；Precision/Boundary 不降；Recall/Dice、小中心 Recall 稳定；低骨龄 false split 不增加。

## Anchor and Simplicity Check

唯一贡献保持为：**训练期发育条件化的局部连续 bridge-risk 负梯度分配**。跨病例骨编号、presence head、rank loss、core BCE、推理期 metadata、hard connectivity threshold 均已删除。首版没有新网络参数，标准 nnU-Net 推理不变。

## Closed Method Definition

### 1. Sidecar source and abstention

原数据的 PNG 标签实际包含图内实例值（0 为背景，1..N 为不同标注实例）；这些数值只作为单图内 identity，不假设跨病例编号语义一致。

sidecar audit 必须拒绝：

- binary connected component 与 instance value 明显不一致；
- 单实例断裂为多个大组件；
- 两实例在 binary label 中已经粘连且没有 y=0；
- 极小伪组件；
- 增强后 instance split/merge；
- pair corridor 穿过第三实例。

### 2. Separate context and distance

定义归一化实际间距：

```text
d_ij = minimum boundary distance / min(local bone width_i, width_j)
```

上下文描述符不包含 d：

```text
u_ij = [
  orientation-normalized pair midpoint,
  relative boundary direction,
  log area ratio,
  compactness pair,
  normalized local ROI coordinate
]
```

发育系数不再声称“异常接近”：

```text
q_context = train-density confidence(a, sex, u_ij)
q_close   = exp(-d_ij / tau_close)
r_dev     = q_ESS(a,sex,u_ij) × q_context × q_close
```

tau_close 以相对骨宽定义并在 train-only Gate A 预注册。骨龄/性别只提供 training-only context density 与 ESS；实际几何距离决定接近风险。每病例 pair 先归一化，防止高骨龄 pair 数更多。

### 3. Fixed local capsule

令 x_i*、x_j* 为两实例最近边界点，w_i、w_j 为局部短轴宽度：

```text
Omega_ij =
  Dilate(line_segment(x_i*, x_j*),
         radius = rho × min(w_i,w_j))
```

预注册约束：

- capsule length <= L_max × min(w_i,w_j)；
- rho、L_max 只用 train audit 确定；
- Omega 与第三实例 support 相交则拒绝 pair；
- source/target eroded cores 从路径内部排除；
- legal anchors = Omega ∩ {y=0}；
- 不允许 capsule 外绕行。

首版只允许直线 capsule，不使用曲线路径。

### 4. Local widest-path bridge risk

在 Omega_ij 的 8-neighbor graph 内，以两个边界邻域为 source/target：

```text
b_ij =
  max over P subset Omega_ij,
           |P| <= L_ij
  min over x in P excluding source/target cores
  detach(p_fg(x))
```

路径不得进入第三实例或离开 capsule。使用 maximin Dijkstra/priority flood 计算；路径长度超过预注册上限则 pair 无效。b_ij 只作 stop-gradient 连续系数。

### 5. X-ray and annotation reliability

局部强度方向由骨内与 gap 相对值确定：

```text
q_valley = clip(
  [min(I_bone_i,I_bone_j)-I_gap]
  / [abs(I_bone_i-I_bone_j)+sigma_local+eps],
  0,1)

q_xray = q_valley × q_double_edge
```

q_double_edge 为两侧方向相反且分别指向两个支持的归一化梯度证据。

完整像素可靠度：

```text
q_x = q_corridor × q_xray × q_aug × q_ann
```

- q_corridor：距 capsule 中轴的连续衰减；
- q_xray：连续 [0,1]；
- q_aug：两次轻微增强后 anchor 存活与位置一致性；
- q_ann：实例/二值一致、非断裂、非第三骨穿越的连续/硬组合。

1 px gap 只有 q_xray 和 q_aug 均通过 train-only hard floor 才保留。loss 只在 full-resolution logits 计算。

pair 系数：

```text
k_ij = stopgrad(r_dev × b_ij)
a_x  = max_{ij:x in pair} k_ij q_x
```

### 6. Gradient-clipped background BCE

令 z = logit_fg - logit_bg，g_max 为最大单像素梯度，z0=logit(g_max)：

```text
ell_gc(z) =
  softplus(z),                              if sigmoid(z) <= g_max
  softplus(z0) + g_max × (z-z0),           otherwise
```

因此严重高置信假桥仍有固定非零纠正梯度 g_max，但不会获得无限/过强权重。

按图像归一化：

```text
L_close_image =
  sum_x a_x ell_gc(z_x) / (eps + sum_x a_x)

L_close = mean_valid_images(L_close_image)
L = L_nnunet_native + alpha_fixed L_close
```

无有效 pair 的图像只使用 native loss。删除低概率停止阈值，避免重新引入阈值抖动。

### 7. One-time alpha calibration

alpha 不做 batch-wise 动态更新。用固定的 train-only calibration manifest（32 张，按骨龄分层但不看 val）测量 full-resolution logits 上的梯度范数：

```text
alpha_fixed =
  clip(0.05 × median(g_native / [g_close+eps]),
       alpha_min, alpha_max)
```

之后冻结。目标是 close-gap 梯度中位数不超过 native 的 5%。训练中梯度比只作监控/停止条件，不反馈调节 alpha。

### 8. Exact protocol

- **Gate A**: train-only non-GPU anchor/corridor audit；不通过不训练。
- **Warm-up**: 2 epoch native nnU-Net，保存共同 checkpoint。
- **Gate B**: 从同一 warm-up checkpoint 分叉：
  - native continuation 2 active epochs；
  - native + close-gap 2 active epochs。
- **Gate C**: Gate B 通过后才做 10–15 epoch original-val pilot。
- clean-test-v2 始终锁定。

Gate A 硬门槛：

- sidecar recoverability；
- third-instance crossing = 0；
- widest-path capsule escape = 0；
- 1 px/2 px anchor augmentation survival；
- X-ray double-edge pass rate；
- 按骨龄的 valid pair、ESS、activation；
- 原始 px、预处理 px、gap/local-width 三种尺度。

Gate B/C guardrails：

- close-gap p_fg 按 1–2、3–4、5–8、>8 px；
- false merge；
- Precision、Boundary IoU/F1；
- per-instance Recall、small-center Recall；
- false split、fragment increase、foreground area loss；
- Dice/Recall non-inferiority。

## Claim Scope

首轮只证明 local continuous bridge-risk anchor 能否安全降低高骨龄近缝假连接。发育条件化的贡献必须由以下三项对照后才能主张：

1. native；
2. native + unconditioned local bridge loss；
3. native + age/sex-conditioned local bridge loss。

后续删除 q_xray 和 b_ij 分别验证图像证据与连续桥风险。正式主张需要多 seed 或 nnU-Net folds。未通过 Gate C 不扩展骨干、不运行 clean-test-v2。

