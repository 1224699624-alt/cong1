# 双结构先验项目：Codex 桌面端最终施工任务书

## 0. 项目目标

当前阶段不再继续扩展新的评价指标，也不再沿 R352/R353 无限增加新机制。目标是把现有 R351 双结构先验框架收敛成可复现、可消融、可更换 backbone、可用于论文最终实验的版本。

核心研究目标：

1. 在 TSRS 和 RAM 两个数据集上证明结构先验有效。
2. TSRS 以骨缝/分离结构先验为主要贡献，RAM 以骨重叠/补全先验为主要贡献。
3. 两个先验保持统一框架，但允许根据监督可用性采用不同损失权重或作用强度。
4. 先在 nnU-Net 上完成严格 matched ablation。
5. 再替换至少两个通用骨干（建议 TransUNet、SwinUMamba）验证 plug-and-play / backbone-agnostic 特性。
6. 最终论文主结果只使用标准分割指标；已有 gap/merge/overlap-region 指标保留为开发诊断或补充材料，不新增自定义主指标。
7. 最终模型在 RAM 官方 benchmark 和 TSRS 的强基线比较中达到有竞争力的结果。

---

## 1. 当前仓库与 R351 状态

仓库：

`1224699624-alt/cong1`

重点文件：

- `scripts/r351_dual_prior.py`
- `scripts/r351_data.py`
- `scripts/train_r351_overlap_prior.py`
- `scripts/train_r351_ram_dual_prior.py`
- `scripts/nnunet_trainers/nnUNetTrainerR351DualPrior.py`
- `scripts/r351_evaluation.py`
- `scripts/summarize_r351_tsrs.py`
- `run_r351_tsrs_dual_prior.sh`
- `run_r351_ram_dual_prior.sh`
- `research-workflow/refine-logs/R351_BACKBONE_CONTRACT.md`
- `research-workflow/refine-logs/R351_EVALUATION_NOTES.md`
- `research-workflow/refine-logs/R351_TRACKER.md`
- `outputs/artifact_bundles/r351_20260917/`

当前已确认：

- RAM overlap prior 使用真实 multi-label overlap target：`mask.sum(1) >= 2`。
- TSRS 没有把前景并集伪造成真实 overlap GT；`membership_observed=False` 时 overlap loss 被屏蔽。
- RAM 固定 seam ROI 和 checkpoint/epoch/hash 绑定已经完成。
- 当前 TSRS R351 由 R350 mature 初始化，而非真正 native-start。
- 当前 RAM R351 由 R332 mature 初始化，而非真正 native-start。
- 当前 TSRS R351 相对 native 标准指标总体正向，但相对 R350 基本持平，因果贡献尚不干净。
- 当前 TSRS trainer 冻结 backbone，只训练输入/输出 prior adapter。
- 当前 TSRS overlap prior 是从 RAM 迁移的弱证据，没有 TSRS 真 overlap supervision。
- 不要删除或覆盖历史 R317/R350/R351 结果。

---

## 2. 必须先解决的问题

### P0. TSRS 实例标签解析风险（最高优先级）

现有旧代码中至少存在：

`train_r256_scale_invariant_pair_prior.py::read_instance`

`train_r258b_prediction_relation_prior.py::read_instance`

其逻辑类似：

```python
value = np.asarray(Image.open(path))
return (value[..., 0] if value.ndim == 3 else value).astype(np.int32)
```

如果 TSRS 标签是 RGB/RGBA 彩色 mask，只取 R 通道可能把不同 RGB 颜色错误合并成同一实例。

TSRS 特点：

- 儿童年龄不同，不存在固定 27 实例假设。
- 不同图片颜色与骨头身份没有稳定对应关系。
- 颜色只允许作为“单张图内匿名实例标记”。
- 少数图片存在两块骨使用相同颜色的问题。
- 同色但空间不相连时，可以通过 connected components 自动拆开。
- 同色且相连时不能可靠自动恢复，不允许硬猜。

#### 施工要求

新增共享模块，例如：

`scripts/tsrs_instance_labels.py`

至少实现：

```python
decode_anonymous_instances(path) -> np.ndarray[int32]
audit_instance_label(path) -> dict
```

解析原则：

1. `P/L/I` 等单通道索引图：
   - 每个非背景 label 再做 connected-component splitting。
2. `RGB/RGBA`：
   - 使用完整 RGB tuple，而不是单独 R 通道。
   - 对每个非背景 RGB tuple 再做 connected-component splitting。
3. 每张图从 1 重新编号，跨图不要求 ID 一致。
4. 不假定实例数量为 27。
5. 不尝试建立固定骨类别。
6. 主二值分割 GT 仍定义为 `instance_id > 0`。
7. 对无法自动判断的同色相连标注，不进行激进修复，只记录 QC。

增加 dataset audit：

- 图像模式分布：P/L/RGB/RGBA 等。
- unique color/index 数量分布。
- anonymous instance 数量分布。
- 同色多 connected-components 图片数量。
- first-channel collision 数量：
  - 不同 RGB tuple 具有相同 R 值的情况。
- 可恢复重复颜色实例数量。
- 输出 `json + csv`。

#### Go/No-Go

- 如果全数据均为可靠 P/indexed palette，且旧读取不会丢实例信息：不因本问题重训历史 seam prior。
- 如果存在 RGB/RGBA 且 first-channel collision：修复 decoder，并重新生成与 TSRS seam prior 有关的训练样本/先验。
- 不修改标准评价指标。

---

### P1. 最终实验不能继续使用 R350/R332 作为唯一初始化

当前 R351 是成熟模型上的再微调，因此不能干净证明 R351 prior 本身的贡献。

最终论文实验必须建立 matched native-start 版本。

TSRS native anchor：

- 优先使用 tracker 中冻结的 native/matched nnU-Net checkpoint。
- 当前记录：
  `outputs/nnunet/r317_image_centers_only/r202_two_channel_zero_prior.pth`
- 使用前必须核对实际存在、SHA、输入契约，并复现当前 native R201 标准指标。

RAM native anchor：

- `outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth`
- tracker SHA256：
  `88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9`

#### 施工要求

新增 final-run 配置，不修改历史 R351：

- `final_native_continue`
- `final_seam_only`
- `final_overlap_only`
- `final_dual_prior`

四组必须：

- 同一数据 split
- 同一 native checkpoint
- 同一随机种子
- 同一训练轮数
- 同一 optimizer/scheduler
- 同一 augmentation
- 同一 checkpoint selection rule
- 同一 evaluator

R350/R332 只保留为历史 mature anchor/reference，不能作为最终因果消融的 baseline。

---

### P2. 解决“额外训练轮数”这个混杂因素

不能比较：

`native checkpoint（未继续训练）` vs `+prior（又训练了若干 epoch）`

否则审稿人会质疑增益来自额外训练。

#### 施工要求

增加 matched control：

`native_continue`

含义：

- 从与 +prior 完全相同的 native checkpoint 开始。
- 训练相同 epoch/iteration。
- 相同 LR schedule。
- 不启用 prior adapter/refiner。
- 最终 base 与 +prior 都从自己的同轮次最佳 checkpoint 评价。

最终真正用于论文的比较：

```text
Native-Continue
vs
Native + Seam Prior
vs
Native + Overlap Prior
vs
Native + Dual Prior
```

原始 frozen native checkpoint 只作为额外参考。

---

### P3. TSRS 上 overlap branch 必须保持保守

TSRS 没有真实 overlap supervision。

当前：

- overlap prior 来自 RAM。
- `membership_observed=False` 已经正确关闭 overlap strong loss。
- 但 output refiner 仍可能通过 base/separation/preserve 梯度改变结果。

这可能抵消 seam prior 的收益。

#### 施工要求

不要立刻发明新网络。

先给现有模块增加明确、可配置的 residual strength / branch enable：

```text
enable_input_prior
enable_output_prior
output_residual_scale
overlap_context_scale
```

要求：

- 默认行为必须能完全复现现有 R351。
- `seam_only`：
  - 输入 seam prior 开启。
  - 输出 completion residual 关闭或 residual_scale=0。
- `overlap_only`：
  - 输入 seam adapter 关闭。
  - 输出 prior 开启。
- `dual`：
  - 两者都开。
- TSRS 上 strong overlap loss 永远保持 disabled，除非以后真的获得 multi-membership GT。

不要在正式 test 上搜索 residual scale。

如果 dual 在 native-start pilot 中比 seam-only 差：
- 优先降低 TSRS output residual strength；
- 不改变评价指标；
- 不新增复杂 loss。

---

### P4. TSRS backbone 当前完全 frozen，需要区分“证明插件有效”和“追求最终性能”

当前 `nnUNetTrainerR351DualPrior.py` 中：

```python
self.backbone = backbone.requires_grad_(False)
```

这适合隔离 prior 的增益，但可能限制最终 SOTA 性能。

#### 施工要求

支持两个明确模式：

### Mode A: `adapter_only`

- backbone frozen
- 只训练 prior 模块
- 用于快速 pilot 和证明 plug-in 本身能改变性能

### Mode B: `joint_finetune`

- backbone 可训练
- prior 与 backbone 联合微调
- backbone 使用更小 LR，例如 adapter LR 的 0.1~0.25 倍
- matched baseline `native_continue` 必须使用相同 backbone LR、epoch、scheduler
- 保留现有 preserve/no-harm 逻辑
- 不新增复杂损失

运行顺序：

1. 先 adapter_only pilot。
2. 如果方向正确，再跑 joint_finetune。
3. 论文主表优先使用公平的 joint_finetune matched result。
4. adapter_only 可以作为 plug-in efficiency 补充实验。

---

### P5. TSRS seam prior 如果仍然弱，再做“腕骨区域聚焦”，但不改指标

这不是第一步。

只有在以下情况发生时才启动：

- label decoder 已确认正确/修复；
- native-start seam-only 仍几乎无收益；
- dual 的问题已排除；
- 标准 Boundary/Surface 指标仍无明显改善。

仓库已有可复用代码：

- `build_r265_hierarchical_carpal_priors.py`
- `run_r269_wrist_internal_merge_roi_pilot.py`

TSRS 研究重点是儿童腕骨骨化中心附近 seam，而不是所有指骨骨骺。

如果需要优化 seam prior：

- 复用现有 wrist/carpal localization 逻辑。
- 仅过滤或加权 seam prior 候选。
- 不新增论文主指标。
- 仍使用现有标准 Dice/IoU/Boundary/Surface 评价。
- 不假设 27 块骨。
- 优先使用相对尺度/图像位置，而不是固定像素或固定实例数量。

---

## 3. 不要修改的内容

以下内容视为冻结协议：

### TSRS 主评价

继续使用现有 R201 中的标准指标：

- Dice
- IoU
- Precision
- Recall
- Boundary IoU
- Boundary F1
- Surface Dice 2 px
- Surface Dice 5 px
- HD95
- ASSD

已有：

- gap_region_fp_rate
- component_merge_rate
- component_count_mae

保留为开发诊断或 supplementary，不新增新的 paper-facing 自定义指标。

### RAM 主评价

优先对齐 RAM 官方 benchmark 标准指标：

- DSC
- NSD
- VOE
- MSD
- RAVD

现有 overlap/pair metrics 可以继续作为内部机制诊断，但不要让论文结论依赖新自定义指标。

### 数据划分

- 不使用 test 做调参。
- pilot / model selection 仅 train + val。
- 最终超参数冻结后才打开 test。
- 现有 test 防护逻辑继续保留。

### 结果完整性

继续保留 R351 已有的：

- checkpoint epoch 绑定
- checkpoint SHA256
- evaluator version
- config
- seed
- result manifest
- 禁止覆盖已完成 run

---

## 4. R351 代码重构目标

不要重写整个项目。

在现有 `scripts/r351_dual_prior.py` 基础上抽象出统一插件接口。

建议：

```python
class StructuralPriorPlugin(nn.Module):
    def __init__(
        self,
        seam_prior,
        overlap_prior,
        input_adapter,
        output_refiner,
        enable_input=True,
        enable_output=True,
        output_residual_scale=1.0,
    ):
        ...

    def adapt_input(self, image, seam):
        ...

    def refine_output(self, image, logits, seam):
        ...
```

backbone contract：

```python
class BackboneAdapter(nn.Module):
    def forward_logits(self, image_or_input) -> torch.Tensor:
        ...
```

要求：

- prior 模块不能访问特定 encoder/decoder 中间层。
- prior 模块不能依赖 nnU-Net 层名称。
- 允许不同任务做 input/output representation conversion。
- 所有 backbone 的最终接口统一为 logits。
- 不能让 GT 进入 inference forward。

保留现有：

- `ImageOverlapPrior`
- `SeamInputAdapter`
- `OutputCompletionPrior`
- `local_terms`
- gradient routing
- preserve/no-harm
- identity initialization

优先做“薄 wrapper”，不要重构成庞大框架。

---

## 5. nnU-Net 最终消融矩阵

两个数据集都至少跑：

| ID | Backbone | Input Seam Prior | Output Overlap Prior | Purpose |
|---|---|---:|---:|---|
| A | nnU-Net | × | × | matched continue baseline |
| B | nnU-Net | ✓ | × | seam prior effect |
| C | nnU-Net | × | ✓ | overlap prior effect |
| D | nnU-Net | ✓ | ✓ | full framework |

### TSRS 预期

不要求 overlap-only 显著提升。

最低合理目标：

- B 相对 A：
  - Dice/IoU 不明显退化；
  - Boundary/Surface 标准指标出现一致改善。
- D 相对 A：
  - 至少保持整体性能；
  - 最好接近或超过 B。
- 如果 B 好、D 差：
  - 优先降低 TSRS output prior residual strength；
  - 不推翻 seam prior。

### RAM 预期

最低合理目标：

- C 相对 A：
  - 官方标准指标改善或至少整体不退；
  - overlap/pair internal diagnostics 应显示方向一致。
- D 相对 A：
  - 最好达到整体最优或最稳定。

---

## 6. Pilot -> Full 的运行策略

不要一开始就跑长训练。

### Stage 0: 静态审计

- TSRS label mode + RGB collision audit。
- 所有 checkpoint/path/hash 检查。
- prior map coverage 检查。
- train/val identity overlap 检查。
- 单元测试。
- 真实 batch identity test。

### Stage 1: 2~3 epoch / 短 iteration matched pilot

TSRS：

- A native_continue
- B seam_only
- D dual

RAM：

- A native_continue
- C overlap_only
- D dual

先不跑全部大矩阵。

### Stage 2: Go/No-Go

TSRS pilot go 条件建议：

- Dice 与 IoU 不出现明显负向；
- Boundary IoU / Boundary F1 / Surface Dice / HD95 / ASSD 中至少多数方向一致改善；
- seam-only 至少能证明有稳定正向信号。

RAM pilot go 条件：

- Overall DSC / NSD / VOE / MSD / RAVD 无系统性退化；
- overlap internal metrics 至少有清晰改善。

### Stage 3: 完整 nnU-Net A/B/C/D

只有 pilot 通过后跑完整训练。

### Stage 4: 3 seeds

第一轮只用 1 seed 找实现问题。

最终进入论文的 selected configuration 至少 3 seeds，报告 mean ± std 或 bootstrap CI。

---

## 7. Backbone 替换验证

nnU-Net 稳定以后再做。

建议：

1. nnU-Net
2. TransUNet
3. SwinUMamba

理由：

- 三者结构差异足够大。
- RAM 官方 benchmark 已包含相关模型，实现容易对齐。
- 足以支撑“不是只对 nnU-Net 有效”。

### 每个新 backbone 只跑两组

```text
Backbone Base
Backbone + Full Dual Prior
```

不需要在每个 backbone 上重新跑 A/B/C/D 全消融。

A/B/C/D 只在 nnU-Net 做完整机制消融即可。

### 要求

- 优先复用 RAM-W600 官方实现或仓库中已有实现。
- 不自己重新实现 TransUNet/SwinUMamba 核心网络。
- base 和 +prior 使用相同：
  - 数据
  - preprocessing
  - augment
  - optimizer
  - epoch
  - LR schedule
  - seed
  - checkpoint selection
- prior 部分保持同一份 `StructuralPriorPlugin` 代码。

---

## 8. RAM SOTA 竞争路线

RAM 最终必须按官方 test protocol 输出：

- DSC
- NSD
- VOE
- MSD
- RAVD

流程：

1. 在 train/val 完成所有选择。
2. 冻结：
   - 模型结构
   - prior 权重
   - loss 权重
   - residual scale
   - threshold
   - checkpoint selection rule
3. 只运行一次正式 test。
4. 自动生成与官方 benchmark 同列的结果表。
5. 不用 test 结果返回调参。

目标不是要求每项绝对第一。

论文可以支持的强结论顺序：

1. competitive with strong/SOTA architectures
2. outperforms strong baselines
3. 若多个官方指标确实超过公开最佳，再讨论 stronger SOTA wording

禁止在结果未完成前预先写“new SOTA”。

---

## 9. TSRS 最终路线

TSRS 不要强行寻找不可比的外部 SOTA 数字。

最终主表：

- nnU-Net
- TransUNet
- SwinUMamba
- 对应 + Prior

全部使用相同 TSRS split 和相同标准 evaluator。

如果 +prior 在多个 backbone 上都稳定提高标准指标，即足以支撑：

`the structural prior is backbone-agnostic / plug-and-play`

TSRS 不要求 Dice 大幅上涨。

对结构先验而言，以下结果可以接受：

- Dice/IoU 小幅提高或近似持平；
- Boundary IoU/F1、Surface Dice、HD95、ASSD 有更清晰改善；
- 这种改善在至少两种 backbone 上复现。

---

## 10. 结果目录与 run manifest

新增隔离目录，不覆盖 `/root/r351_workspace`。

建议：

```text
outputs/final_prior_v1/
  tsrs/
    nnunet/
      native_continue/
      seam_only/
      overlap_only/
      dual/
    transunet/
      base/
      dual/
    swinumamba/
      base/
      dual/

  ram/
    nnunet/
      native_continue/
      seam_only/
      overlap_only/
      dual/
    transunet/
      base/
      dual/
    swinumamba/
      base/
      dual/
```

每个 run 必须有：

`run_manifest.json`

至少记录：

```json
{
  "dataset": "...",
  "split": "...",
  "backbone": "...",
  "variant": "...",
  "seed": 0,
  "git_commit": "...",
  "source_checkpoint": "...",
  "source_checkpoint_sha256": "...",
  "selected_checkpoint": "...",
  "selected_checkpoint_sha256": "...",
  "selected_epoch": 0,
  "training_mode": "adapter_only|joint_finetune",
  "config": {},
  "evaluation_protocol": "...",
  "test_used_for_selection": false
}
```

---

## 11. 单元测试和防回归测试

至少增加：

1. `test_tsrs_instance_decode_palette`
2. `test_tsrs_instance_decode_rgb_full_tuple`
3. `test_tsrs_same_color_disconnected_components_split`
4. `test_binary_union_unchanged_after_instance_remap`
5. `test_no_gt_enters_prior_forward`
6. `test_seam_only_disables_output_residual`
7. `test_overlap_only_disables_input_adapter`
8. `test_dual_identity_initialization`
9. `test_native_continue_has_no_prior_parameters_active`
10. `test_same_checkpoint_epoch_hash_binding`
11. `test_test_split_not_used_for_selection`
12. `test_backbone_adapter_output_shape`

真实 batch smoke test 必须在 full run 之前通过。

---

## 12. Codex 第一轮需要交付的内容

第一轮不要直接启动长实验。

先交付：

### A. Repository audit report

`research-workflow/refine-logs/FINAL_PRIOR_IMPLEMENTATION_AUDIT.md`

包含：

- TSRS 标签真实 mode 分布。
- 是否存在 RGB first-channel collision。
- 是否需要重建 seam prior。
- 当前 native checkpoints 是否完整。
- current R351 -> final architecture 的修改列表。
- 预计需要重新训练哪些部分。
- 哪些历史结果可以直接复用。

### B. Code patch

完成：

- TSRS label decoder + audit。
- R351 enable/disable branch 配置。
- native_continue / seam_only / overlap_only / dual 配置入口。
- adapter_only / joint_finetune 模式。
- run manifest。
- 必要单元测试。

### C. Dry run

只做：

- synthetic tests
- real-batch smoke test
- 1~2 case inference
- 不启动长训练

### D. Pilot commands

输出可直接执行的：

- TSRS 3 个 pilot 命令：A/B/D
- RAM 3 个 pilot 命令：A/C/D

等待人工确认后再跑。

---

## 13. Codex 的停止条件

遇到以下情况必须停止并汇报，不要自行继续堆新方法：

1. TSRS label 实际格式与当前假设不符。
2. 修复 decoder 后 prior map coverage 不完整。
3. native checkpoint 无法复现既有 baseline。
4. A/B/C/D 不是从完全相同 native checkpoint 开始。
5. test split 被任何训练/阈值选择读取。
6. dual pilot 比 native 明显下降且 seam-only 也下降。
7. 发现需要改变论文评价协议才能“让结果变好”。
8. 发现必须删除/覆盖历史结果才能继续。
9. 任何 checkpoint/epoch/hash 对不上。

停止后只提交诊断，不自动发明 R352/R353。

---

## 14. 明确禁止事项

Codex 不要做：

- 不重新设计论文主评价指标。
- 不引入新的自定义 paper-facing metric。
- 不假设 TSRS 固定 27 块骨。
- 不把颜色当跨图固定骨类别。
- 不把 TSRS binary foreground 当真实 overlap GT。
- 不用 RAM test 或 TSRS final test 调参。
- 不从 R350/R332 开始做最终因果消融。
- 不删除历史 R317/R350/R351。
- 不把 pilot 结果包装成最终 test 结果。
- 不因为某个结果不好就自动换阈值、换 split、挑 epoch 拼表。
- 不在多个 epoch 之间拼接“最好指标”。
- 不为了追求 SOTA 改 evaluator。

---

## 15. 最终论文需要得到的三类证据

### Table 1: Prior Ablation on nnU-Net

TSRS + RAM：

```text
Native Continue
+ Seam Prior
+ Overlap Prior
+ Dual Prior
```

证明两个 prior 的贡献和组合效果。

### Table 2: Backbone Portability

TSRS + RAM：

```text
nnU-Net
nnU-Net + Prior
TransUNet
TransUNet + Prior
SwinUMamba
SwinUMamba + Prior
```

证明 prior 不依赖单一 backbone。

### Table 3: Competitive Comparison

RAM：

按官方 benchmark 的：

```text
DSC / NSD / VOE / MSD / RAVD
```

与公开强模型公平比较。

TSRS：

在完全相同 split/evaluator 下，与 nnU-Net、TransUNet、SwinUMamba 等强基线比较。

---

## 16. 当前最重要的执行顺序

严格按下面顺序：

```text
1. TSRS label-format audit
        ↓
2. 修复 decoder（仅在确认有问题时）
        ↓
3. 确认/复现 native baseline
        ↓
4. 重构 R351 branch switches + matched controls
        ↓
5. 单元测试 + real-batch smoke
        ↓
6. TSRS A/B/D short pilot
        ↓
7. RAM A/C/D short pilot
        ↓
8. 根据标准指标决定是否进入 full nnU-Net A/B/C/D
        ↓
9. 3-seed final nnU-Net
        ↓
10. TransUNet / SwinUMamba base vs +prior
        ↓
11. 冻结全部配置
        ↓
12. 最终 test
        ↓
13. 自动生成论文表格
```

---

## 17. 项目的核心判断标准

不要以“Dice 必须大涨”作为唯一成功标准。

TSRS 成功：

- Dice/IoU 不明显退化；
- Boundary/Surface 标准指标出现稳定改善；
- +prior 在多个 backbone 上可复现。

RAM 成功：

- 官方标准指标整体有竞争力；
- overlap prior 对强 backbone 仍能提供增益；
- 最终 test 结果与公开 benchmark 具有竞争性。

最终要证明的是：

```text
Structural Prior Plugin
        +
Different Generic Backbones
        ↓
Consistent or Reproducible Improvement
```

而不是证明某一个历史 R351 checkpoint 本身就是最终模型。
