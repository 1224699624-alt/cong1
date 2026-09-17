# 下一轮主候选修补方案

**日期**: 2026-06-15 22:22  
**依据**:
- `research-workflow/refine-logs/EXPERIMENT_RESULTS.md`
- `research-workflow/refine-logs/EXPERIMENT_PLAN.md`
- 远端 hard-case 对比目录  
  `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/bridge_hardcase/aic_vs_keepbone_cutgap_v2`

## 问题锚点

当前项目的真实瓶颈不是粗定位，而是：

- 薄骨骺区域不能漏掉
- 相邻骨之间的窄缝不能被错误连上
- 边界完整性提升不能靠牺牲 precision 换来

下一轮修补必须继续服务这个锚点，不能为了保守而回到“边界不够、召回不足”的旧状态。

## 本轮失败诊断

## 整体指标结论

`allstack_anatomy_roi_aic_refiner_sam_hqsam` 的 profile 很稳定：

- clean-test-v2 Dice `0.905869`，低于：
  - 当前内部最强 `keepbone_cutgap_v2` 的 `0.910029`
  - ARAA 的 `0.911766`
- 但它同时有：
  - 更高 recall
  - 更高 boundary IoU
  - 明显更低 precision

这说明它学到的不是“错误方向”，而是“缺少精度回收器的高召回方向”。

## 1433 / 1475 的局部失效模式

这两个样本是下一轮必须优先修的 hard-negative case。

### `1433`

- keepbone-cutgap-v2:
  - Dice `0.8503`
  - boundary IoU `0.1755`
- aic-refiner:
  - Dice `0.8263`
  - boundary IoU `0.1501`

结论：
- 不是单纯少了一点边界，而是 `aic_refiner` 在这个样本上把错误前景也带进来了
- 这类 case 需要更强的“假阳性 veto”

### `1475`

- keepbone-cutgap-v2:
  - Dice `0.7835`
  - boundary IoU `0.1708`
- aic-refiner:
  - Dice `0.7221`
  - boundary IoU `0.1084`

结论：
- 这是最明显的 precision 崩塌样本
- 说明当前 recover 逻辑会在弱结构证据区域过度补前景

## 2982 / 3040 的正向信号

不能因为整体 verdict 是 `REJECT` 就把 `aic_refiner` 全盘扔掉。

### `2982`

- Dice 仅略低于 keepbone-cutgap-v2
- boundary IoU 更高

### `3040`

- Dice 更高
- boundary IoU 更高

结论：
- `aic_refiner` 的“追回薄结构 + 补边界”机制在一部分 gap-sensitive case 上是有效的
- 下一轮不该抛弃这条信号，而是应该把它限制在“高置信缺失前景区”里使用

## 代码级原因判断

## `aic_refiner` 的核心短板

在 [train_error_refiner.py](/g:/gutou/YOLO+SAM/scripts/train_error_refiner.py:1249) 附近，`allstack_anatomy_roi_aic_refiner` 的 recover 逻辑本质上是：

- 用 `anatomy_gradient / anatomy_band / structure_prior`
- 配合 `boundary_fg_logits`
- 对 `(1 - allstack_prob)` 区域做补回

但它缺少两个关键限制：

1. **候选支持约束不够强**
2. **instance-separation / gap-veto 不参与 recover 决策**

所以它容易在该补的地方补回来，也容易在不该补的地方一起长出来。

## `keepbone_cutgap_v2` 的核心优势

在 [train_error_refiner.py](/g:/gutou/YOLO+SAM/scripts/train_error_refiner.py:1019) 的 `two_stage_keepbone_cutgap_correct(...)` 里，当前最强线已经具备：

- `instance_core_prob`
- `instance_sep_prob`
- `candidate_overlap_gap`
- `disagreement_boundary`
- `prediction_aware` keep/gap stage2

也就是说，`keepbone_cutgap_v2` 已经有更强的 precision 保护结构，只是 recall / 边界追回不如 `aic_refiner` 那么激进。

## 下一轮主候选的总体路线

### 结论

下一轮**不要**继续把 `allstack_anatomy_roi_aic_refiner` 当成独立主候选修。

下一轮主候选应该改成：

`allstack_anatomy_roi_keepbone_cutgap_refiner_v3`

它的定位是：

- **骨架沿用** `keepbone_cutgap_v2`
- **只吸收** `aic_refiner` 的“高召回边界追回”信号
- **不允许** AIC 风格 recover 全局开放

一句话方法 thesis：

> 用 `keepbone_cutgap_v2` 作为 precision-safe 主干，只在“候选缺失但结构证据强、且 gap 风险低”的局部区域，注入 AIC 风格的前景追回。

## 最小充分机制

下一轮只做一个 dominant contribution：

- **candidate-supported precision-rescued recovery**

不要并行再加新分支、新 backbone、新数据清洗主线。

## 建议的最小代码改动

## 改动 1：只在 `keepbone_cutgap_v2` 的 stage2 内注入 AIC 风格 keep cue

修改位置：
- [train_error_refiner.py](/g:/gutou/YOLO+SAM/scripts/train_error_refiner.py:1019)
- 函数：`two_stage_keepbone_cutgap_correct(...)`

建议：

- 新增一个 `aic_keep_prior`
- 组成建议：
  - `anatomy_gradient`
  - `anatomy_band`
  - `structure_prior`
  - `instance_core_prob`
  - `(1 - candidate_union)`

目标：
- 只把 AIC 的“缺失前景恢复能力”作为 `keep_focus` 的补充项
- 不再像 `aic_refiner` 一样直接在全图 recover

推荐形式：

```text
aic_keep_prior = high_structure * missing_candidate * low_gap_risk
keep_focus = max(existing_keep_focus, active_band * aic_keep_prior)
```

关键约束：
- 必须乘上 `low_gap_risk`
- 必须保留 `candidate_union` 缺失条件

## 改动 2：增加 precision veto，专门抑制 1433 / 1475 这类假阳性扩张

修改位置：
- 同样在 `two_stage_keepbone_cutgap_correct(...)`

建议新增一个更强的 `precision_veto`：

- 组成建议：
  - `candidate_overlap_gap`
  - `disagreement_boundary`
  - `instance_sep_prob`
  - `boundary_bg_prob`
  - `(1 - structure_prior)`

目标：
- 对“结构证据不强、但 coarse_prob 想往外长”的区域进行额外 trim

推荐形式：

```text
precision_veto = overlap_gap_or_boundary_disagreement
               * instance_sep_prob
               * bg_support
               * low_structure

corrected_prob = corrected_prob * (1 - extra_trim)
```

这里的 extra trim 应该只作用在：

- `coarse_prob` 较高
- `candidate_union` 不支持
- `gap_focus` 已经活跃

的区域，避免把 `3040` 这类有益恢复也一起削掉。

## 改动 3：训练损失优先增强 gap / trim，而不是继续放大 keep

当前 `keepbone_cutgap_v2` 的关键权重是：

- `--stage2-boundary-loss-weight 0.32`
- `--structure-keep-loss-weight 1.10`
- `--structure-gap-loss-weight 1.20`
- `--boundary-fg-weight 0.55`
- `--boundary-bg-weight 0.58`

下一轮建议先做**小幅偏精度侧**调整：

- `structure_keep_loss_weight`: `1.10 -> 1.00`
- `structure_gap_loss_weight`: `1.20 -> 1.35`
- `boundary_bg_weight`: `0.58 -> 0.62`
- `boundary_fg_weight`: `0.55 -> 0.52`

原则：
- 不追求更强 recover
- 追求“recover 只在该开的区域开”

## 改动 4：threshold sweep 往保 precision 的方向扩一点

当前很多线都在 `0.42~0.54` 这种范围内选阈值。

下一轮主候选建议测试：

- `0.48,0.50,0.52,0.54,0.56,0.58`

原因：
- 这轮 failure 不是“全局漏太多”
- 而是“补太开”
- 所以需要给新候选一个更偏 precision 的 selection 区间

## 下一轮主候选命名建议

主候选：

- `allstack_anatomy_roi_keepbone_cutgap_refiner_v3_sam_hqsam`

如果只做参数微调、不改结构：

- `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam`

命名原则：
- `v3` 留给真正加了 precision-rescued recovery 的版本
- 纯调参版不要冒充新方法

## 下一轮实验块

## Block 1：低风险参数修补

目标：
- 先验证 precision 崩塌是否能靠更偏 trim 的设置压住

系统：
- baseline: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`
- candidate: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam`

成功条件：
- `1433 / 1475` 不恶化
- clean-test-v2 Dice 不低于 `0.910029`

## Block 2：真正主候选 v3

目标：
- 在不放弃 `3040/2982` 这类 recall / boundary 优势的前提下，修复 1433/1475 的 precision 问题

系统：
- `allstack_anatomy_roi_keepbone_cutgap_refiner_v3_sam_hqsam`

成功条件：
- clean-test-v2 Dice `>= 0.9105`
- `1433 / 1475` 至少不低于 keepbone-cutgap-v2
- `3040` 尽量保住当前 AIC 带来的增益

## Block 3：是否继续 AIC 纯线

结论先写清楚：

- 不建议再单独投入 `allstack_anatomy_roi_aic_refiner` 纯线
- 它现在更适合作为 cue source，而不是主候选

## 具体实施顺序

1. 复制 `keepbone_cutgap_v2` launcher，先做一版纯参数修补
2. 在 `two_stage_keepbone_cutgap_correct(...)` 里加入 `aic_keep_prior + precision_veto`
3. 新建 `v3` launcher 和 checkpoint/output 命名
4. 先跑原始 test
5. 再跑 `clean-test-v2`
6. 最后只看四个 hard-case：
   - `1433`
   - `1475`
   - `2982`
   - `3040`

## 不该做的事

- 不要继续单独加大全局 AIC recover
- 不要在这轮就引入 filtered-data 变量
- 不要同时再开两三个结构分支，避免结论失焦

## 最终一句话修补策略

下一轮不再修“纯 `aic_refiner`”，而是把它的**高召回边界追回能力**压缩成 `keepbone_cutgap_v2` 的一个**局部 keep cue**，同时引入更强的 **precision veto / gap trim**，优先修复 `1433/1475` 这类假阳性外扩样本。
