# R270 boundary-classifier pilot

日期：2026-07-20

## 目标

验证“候选 foreground-foreground boundary -> 轻量边界分类器”是否能把真实
相邻结构边界和同一骨内部伪边界分开。训练候选来自独立 train 图像和 train
anchor；original-val 只用于最终 audit。未使用 clean-test-v2，未编辑 mask。

## 实现

- 候选生成：`scripts/run_r269_wrist_internal_merge_roi_pilot.py`
- 分类器：`scripts/train_r270_boundary_classifier.py`
- 特征：neck、distance saddle、X-ray gray valley、two-side support、
  basin balance、combined score、minimum peak thickness
- 模型：24 hidden-unit MLP，weighted BCE，CUDA
- train pilot：96 张 train 图，139 个候选；正边界 6 个
- original-val：160 个可审计候选；正边界 10 个

## original-val 结果

| threshold | precision | recall | selected |
|---:|---:|---:|---:|
| 0.50 | 0.250 | 0.600 | 24 |
| 0.60 | 0.278 | 0.500 | 18 |
| 0.70 | 0.333 | 0.400 | 12 |
| 0.80 | 0.333 | 0.200 | 6 |

## 结论

边界分类比直接按几何分数硬切更合理，但当前结果不满足安全 gate。主要瓶颈：

1. train pilot 只有 6 个正候选，类别极端稀缺；
2. 输入来自二值 anchor，没有成熟模型的 soft probability valley；
3. R269 watershed seeds 经常在同一骨内部产生多个 basin；
4. 边界分类器只能筛已有候选，无法补回候选生成阶段漏掉的真实骨缝。

因此暂不把 R270 结果接入 nnU-Net mask editor，也不启动“先验+损失”长训。
下一步需先用成熟 nnU-Net 保存 train OOF/original-val softmax，再按多阈值概率树生成
候选 foreground-foreground boundaries；只有 boundary precision/recall 明显提高后，
才将可信边界热图作为先验，并加入绝对 gap upper-bound + two-side support loss。
