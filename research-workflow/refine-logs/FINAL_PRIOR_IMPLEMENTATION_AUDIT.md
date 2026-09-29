# Final Prior 第一阶段审计：STOP / 数据不完整

审计日期：2026-09-19。依据用户请求及 Downloads/Codex_Final_Prior_Project_Spec.md。本文仅为停止诊断，不代表第一阶段验收通过。

## 停止原因

本地 `data/raw/TSRS_RSNA-Epiphysis/train_labels` 只有 874 张 PNG，缺少 `1833.png`；val_labels 为 96 张。R351_TRACKER.md 明确记录实验使用远程 875/96 split。不得将本地 874 张替代既定 875 张训练集。按用户“发现与任务书假设不一致时先停止并汇报”要求，停止重构、测试和 pilot 入口生成，等待恢复完整实验数据来源。没有启动训练或正式 test，没有修改历史结果。

## 仓库与文件职责

- 指定仓库：https://github.com/1224699624-alt/cong1 。已读取仓库页面并通过 git ls-remote 核对 HEAD。
- GitHub HEAD 与本地 HEAD 均为 `737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353`。
- 本地 origin 指向 `https://github.com/1224699624-alt/nnunet.git`，未擅自更改。HEAD 相同不意味着未跟踪数据、权重与远程运行环境完整相同。
- 本地有大量既有未跟踪资产；没有移动、清理或覆盖。
- `scripts/r351_dual_prior.py`、`r351_data.py`：现有共享模块与数据入口（停止前未完成逐行审计）。
- `scripts/nnunet_trainers/nnUNetTrainerR351DualPrior.py`：TSRS trainer。
- `scripts/train_r351_ram_dual_prior.py`：RAM trainer。
- `run_r351_tsrs_dual_prior.sh`、`run_r351_ram_dual_prior.sh`：现有历史启动入口，分别使用 R350/R332 mature 初始化，不能充当 final native-start 命令。
- `research-workflow/refine-logs/R351_TRACKER.md`：split、native SHA 与历史运行记录。
- `outputs/artifact_bundles/r351_20260917/`：历史归档入口；本轮没有宣称完成归档全量核验。

## 用户要求的四项确认

### 1. TSRS 标签

用 Pillow 逐文件读取本地 train/val PNG 的 mode：train 为 P × 874；val 为 P × 96。没有读取 test。

两个旧 read_instance 函数直接执行 `np.asarray(Image.open(path))`，只有 ndim==3 才取 `[...,0]`。P 模式返回二维索引，因此对已检查的 970 张标签不会执行截取首通道，不存在该操作造成的 RGB tuple 碰撞。此结论不覆盖缺失文件、test 或其他数据变体。

尚未完成每个索引的 connected components、二值并集、重复颜色、prior coverage、train/val identity 全量审计，不能将 mode 检查称为完整 label audit。目前没有因首通道问题重训 seam prior 的证据。

### 2. Native checkpoints

以下文件存在且 SHA256 与 tracker 一致，路径相对于项目根目录：

| 数据集 | 路径 | 实测 SHA256 |
|---|---|---|
| TSRS | outputs/nnunet/r317_image_centers_only/r202_two_channel_zero_prior.pth | 55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b |
| RAM | outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth | 88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9 |

这是候选 native anchor 的文件完整性确认，不是最终采用验收。输入契约、checkpoint 内部 epoch、真实 batch 与 evaluator 复现均未运行；baseline 是否复现：未确认。

### 3. A/B/C/D 同源性

未确认，尚未实施 final 配置。历史 TSRS launcher 显式将 R351_TSRS_INIT 指向 R350 checkpoint；历史 RAM launcher 传入 --r332-checkpoint。不能将历史入口改名后声称四组从 native 同源初始化。恢复数据一致性后，才进入薄 wrapper、branch switches、matched controls 与 manifest 的实现和验证。

另一个需在实现前明确的协议问题：adapter_only 冻结 backbone，而 native_continue 又禁用所有 prior 参数，此时 A 组没有可训练参数，应明确作为 frozen identity reference；若要实际继续训练 backbone，则应使用公平的 joint_finetune matched 模式，不能暗中混用两种模式。

### 4. 六条 pilot 命令

未生成：目前 final native-start 入口尚不存在，数据 gate 未通过。不给出表面可运行但实际使用 mature 初始化或错误 split 的命令。预期交付仍为 TSRS A/B/D 与 RAM A/C/D，每组 2–3 epoch 或短 iteration；生成并验收后等待人工确认才运行。

## 恢复工作的必要条件

取得原实验 875 张训练标签及匹配图像，或可访问完整远程数据来源，核对 split manifest 与文件身份；不能用补造标签、复制其他样本或改成 874 split 绕过。然后继续 label audit → native baseline 复现 → 分支与 matched controls 重构 → 单元测试和真实 batch smoke → 六条 pilot 命令。

历史 R317/R350/R351 仅保留为历史参考，未验证为本轮 matched 因果证据。没有新增 R352/R353，没有运行长训练，没有改变指标或数据。

## 用户确认后的继续执行结果（2026-09-19）

用户确认缺失文件可在服务器恢复后，项目不冻结为 874；服务器实际存在 `1833.jpg` 与 `1833.png`，因此最终统一 split 为 TSRS 875 train / 96 val。服务器与本地的 split、图像、标签、prior coverage 审计通过，TSRS split SHA256 为 `21f9272b992bdd17e2ae13aa7bd977f6dada2461c9c0e04e33cbabf1648f29f4`。RAM 425/69 coverage 也通过。详细结果及六条 pilot 见 [FINAL_PRIOR_PILOT_REPORT.md](FINAL_PRIOR_PILOT_REPORT.md)。

native baseline 已在服务器完成真实验证：TSRS fresh R201 Dice `0.9036641175689711`、IoU `0.8276992141832928`、HD95 `16.963342422177195`、ASSD `4.381488342513816`；RAM official overall DSC `0.9788577444266571`、NSD `0.8872615864685836`、VOE `0.040899481692983526`、MSD `1.1692853896336834`、RAVD `0.02026307045041241`。两个 source SHA 与本文件前述值一致。

实现了 `scripts/tsrs_instance_labels.py`、`scripts/final_prior_plugin.py`、`scripts/nnunet_trainers/nnUNetTrainerFinalPrior.py`、`scripts/train_final_prior_ram.py` 及对应入口和测试。TSRS D audit 与 RAM D audit 通过；TSRS A/B/D 与 RAM A/C/D 短 pilot 均完成。TSRS mean validation Dice 为 A `0.9037335`、B `0.9037259`、D `0.9037182`；RAM best val overall DSC 为 A `0.9786964`、C `0.9786078`、D `0.9786541`。这些是短 pilot 诊断，不是最终论文结果；test 未读取，full/3-seed 未启动。
