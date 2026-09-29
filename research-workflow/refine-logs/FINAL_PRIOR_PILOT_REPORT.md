# Final Prior 第一阶段交付与短 pilot 报告

日期：2026-09-19。用户确认服务器存在 `1833.png` 后，统一恢复并固定完整 TSRS 875/96 split；没有启用 874 备选 split。

## 已确认的四件事

TSRS train/val 971 张标签全部是 Pillow `P` 模式；first-channel collision 为 0。P 模式是二维索引数组，旧 `[..., 0]` 分支不会执行，因此没有由 RGB 首通道截取造成的实例碰撞。完整 decoder audit 统计了 118 张存在同一索引的多个断开连通域，decoder 会拆成匿名实例并保持 binary union；单个相连区域不做语义猜测。JSON/CSV 位于 `outputs/final_prior_v1/audit/labels/`。

两个 native source 都经过真实验证：

| 数据集 | source | SHA256 | 复现结果 |
|---|---|---|---|
| TSRS | `outputs/nnunet/r317_image_centers_only/r202_two_channel_zero_prior.pth` | `55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b` | 96 val；Dice 0.9036641，IoU 0.8276992，HD95 16.96334，ASSD 4.38149 |
| RAM | `outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth` | `88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9` | 69 val；DSC 0.9788577，NSD 0.8872616，VOE 0.0408995，MSD 1.1692854，RAVD 0.0202631 |

TSRS native 与历史 matched reference 的 Dice/IoU 差分别为 `8.4e-7`/`1.46e-6`，HD95 一致；记录为跨运行微小数值差，不声称逐位相同。RAM native 各比较指标与历史 native baseline 一致。

A/B/C/D 的 trainer 代码统一从上述 native source strict load。TSRS backbone 的第二输入通道始终置零，seam 只进入 `StructuralPriorPlugin`；R350/R332 不作为 backbone source。RAM A/B/C/D runner 也 strict load 同一 native model；R332 仅允许作为 output refiner 的 identity transfer。所有 pilot 记录 source hash、selected epoch、selected checkpoint hash 与 `test_used=false`。

## 真实 batch 与单元测试

- TSRS D adapter-only smoke：backbone identity max abs difference 0，native prior channel max abs 0，finite gradients，seam/output prior gradient routing 有效，TSRS overlap strong loss disabled。
- RAM D audit：2 个真实 validation batch identity max logit/input delta 均 0，finite backward routing 通过；RAM overlap branch 使用真实 multi-label overlap target。
- 本地通过：`scripts/test_final_ram.py` 6 tests、`scripts/test_final_prior_plugin.py` 4 tests、`scripts/test_r351_dual_prior.py` 6 tests；新增模块均通过 `py_compile`。

## 短 pilot（2 epoch × 100 iteration，seed 3511）

TSRS 使用 nnU-Net EMA foreground Dice 选 checkpoint；下面的数值为每个 pilot 最终验证输出中的 Mean Validation Dice，未读取 test：

| variant | branch | Mean Validation Dice |
|---|---|---:|
| A | native_continue | 0.9037335 |
| B | seam_only | 0.9037259 |
| D | dual | 0.9037182 |

相对于新 native reference，三组均没有明显负向退化；D 略低于 B，但差异很小。该结果只证明代码和短预算管线可运行，不能替代 full training 或最终标准指标表。

RAM 使用官方 evaluator、val overall DSC 选 checkpoint：

| variant | branch | best val overall DSC | best epoch |
|---|---|---:|---:|
| A | native_continue | 0.9786964 | 1 |
| C | overlap_only | 0.9786078 | 1 |
| D | dual | 0.9786541 | 1 |

相对 RAM native 0.9788577，这轮短继续训练本身出现约 0.00016–0.00025 的下降；C 与 D 的相对顺序不能支持最终结论，也没有触发长训练授权。pilot 使用 train/val，未读取 test。

## 命令入口

服务器上可直接使用：

```bash
bash run_final_prior_tsrs.sh audit
bash run_final_prior_tsrs.sh A   # 或 B / D
bash run_final_prior_ram.sh audit
bash run_final_prior_ram.sh baseline
bash run_final_prior_ram.sh A   # 或 C / D
```

这些入口默认 2 epoch × 100 iteration，输出隔离在 `/root/final_prior_v1`；目录非空时拒绝覆盖。正式 full A/B/C/D、3 seeds、backbone 替换和 test 均未启动。

## 当前判断

第一阶段的静态审计、native 复现、代码分支、manifest、单元测试、真实 batch smoke 和要求的六条短 pilot 已完成。TSRS pilot 没有明显退化信号；RAM 短预算的 native_continue、overlap_only、dual 都低于冻结 native，说明需要在进入 full run 前由人工决定是否接受这个短预算行为、调整已允许的训练/selection 设置，或暂缓 RAM full A/B/C/D。没有自行增加 R352/R353。
