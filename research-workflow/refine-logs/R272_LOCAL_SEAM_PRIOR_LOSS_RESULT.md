# R272 local indexed-instance seam prior + separation loss

日期：2026-07-20

## 实验设置

- 环境：本地 `YOLOSAM`，CUDA，RTX 3060 Laptop 6 GB；
- 训练：875 train；验证：96 original-val；
- backbone：现有轻量 U-Net pilot（不是最终 nnU-Net，因本地环境没有 nnunetv2）；
- 输入：X-ray；
- 输出：foreground、support/core、seam、boundary、hover auxiliary heads；
- 监督：R271 indexed-instance seam targets；
- 损失：Dice+BCE + seam loss + absolute gap upper-bound + two-side support；
- clean-test-v2：未使用。

## 训练过程

第 11 epoch 早停，综合评分最佳为第 6 epoch。第 6 epoch original-val：

| 指标 | 数值 |
|---|---:|
| Dice | 0.8524 |
| IoU | 0.7509 |
| Recall | 0.9532 |
| Boundary IoU | 0.3885 |
| component count error | 2.8438 |
| seam IoU | 0.1311 |

第 11 epoch seam IoU 达到 0.2182，但 Dice/Boundary IoU 没有超过第 6 epoch，
说明 seam head 继续拟合会与主分割质量产生竞争。因此保存第 6 epoch，而不是
按 seam IoU 单独选模型。

## 输出

- checkpoint：`outputs/experiments/r272_seam_prior_loss_local_full/best.pt`
- original-val masks/seam maps/可视化：
  `outputs/visualizations/r272_seam_prior_loss_local_original_val`
- 训练结果：`outputs/experiments/r272_seam_prior_loss_local_full/result.json`

## 结论

实例 ID 生成的 seam 监督可以被模型学习，证明“实例边界先验 + 分离损失”方向
在本地训练链路上可行。但当前 seam 是全手范围，指骨关节和腕骨都会响应；还
不能把本结果作为掌骨/腕骨粘连改善的最终结论。下一轮应加入腕骨/掌骨 ROI adapter，
再做同骨架 baseline、仅先验、仅损失、先验+损失四组消融，并最终接入 nnU-Net。
