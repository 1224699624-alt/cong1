# 下一轮主候选修补方案

**日期**: 2026-06-15 22:22

## 这份文件是干什么的

这不是“论文式说明”，而是给我们自己看的行动稿：

- 当前到底坏在哪
- 下一轮应该修哪一类错误
- 先跑哪条低风险线
- 真正的结构候选应该改哪里

## 问题锚点

我们的目标一直没变：

- 薄骨骺不能漏掉
- 相邻骨之间的窄缝不能被错误连上
- 边界更完整不能靠 precision 崩掉换来

所以下一轮任何改动，都必须服务这个锚点。

## 上一轮失败的本质

`aic_refiner` 的问题不是“完全没学到东西”，而是：

- 它能追回边界和 recall
- 但它没有足够强的 precision 回收机制

结果就是：

- `3040` 这类样本能变好
- `1433 / 1475` 这类 hard-negative 会炸

## 必修样本

### `1433`

- 这是 precision 崩坏最典型的样本之一
- 下一轮必须重点修

### `1475`

- 也是典型 hard-negative
- recover 一旦太激进，就会在这里出问题

### `2982`

- 是可保留的正向信号样本
- 说明 recall / boundary 追回不是完全错方向

### `3040`

- 是上一轮最强正向 signal
- 下一轮修 precision 时，不能把这个优势一起裁掉

## 结构判断

### 不再继续单独修 `aic_refiner`

原因：

- 它更适合作为 cue source
- 不适合作为独立主候选继续押注

### 下一轮主候选改成 `keepbone_cutgap_v3`

核心思路：

- 骨架保留 `keepbone_cutgap_v2`
- 只吸收 `aic_refiner` 的局部高召回 keep signal
- 同时加强 precision veto / gap trim

一句话：

> 用 `keepbone_cutgap_v2` 做 precision-safe 主干，只把 AIC 风格的前景追回压缩成局部 keep cue，而不是全局 recover。

## 具体代码改动方向

## 改动 1：只在 stage2 里注入 AIC 风格 keep cue

位置：
- `scripts/train_error_refiner.py`
- `two_stage_keepbone_cutgap_correct(...)`

目标：
- 把 AIC 的长处只留在“应该追回的局部区”
- 不允许它全局往外长

## 改动 2：加入更强的 precision veto

位置：
- 同样在 `two_stage_keepbone_cutgap_correct(...)`

目标：
- 专门压制 `1433 / 1475` 这种假阳性外扩
- trim 只作用在高风险 gap / 低结构可信区域

## 改动 3：loss 权重更偏 precision 侧

当前建议：

- `structure_keep_loss_weight`: `1.10 -> 1.00`
- `structure_gap_loss_weight`: `1.20 -> 1.35`
- `boundary_bg_weight`: `0.58 -> 0.62`
- `boundary_fg_weight`: `0.55 -> 0.52`

原则：

- 不是继续放大 recover
- 而是让 recover 更克制、更精准

## 改动 4：threshold 区间更偏 precision

建议范围：

- `0.48, 0.50, 0.52, 0.54, 0.56, 0.58`

原因：

- 这轮失败不是“漏得太多”
- 而是“补得太开”

## 运行命名

### 纯调参版

- `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam`

### 结构修补版

- `allstack_anatomy_roi_keepbone_cutgap_refiner_v3_sam_hqsam`

## 运行顺序

### 第一步：低风险线

先跑：

- `v2_precision_tune`

目的：

- 看单纯偏 precision 的设置，能不能先把 `1433 / 1475` 压住

### 第二步：结构版主候选

如果低风险版不够，就跑：

- `v3`

目的：

- 在保住 `2982 / 3040` 正向信号的同时
- 真正修掉 `1433 / 1475`

## 每轮看什么

只看这三层：

1. 主切片 `clean-test-v2`
2. 原始 test
3. 四个 hard-case：
   - `1433`
   - `1475`
   - `2982`
   - `3040`

## 这轮不要做什么

- 不要再单独放大全局 AIC recover
- 不要现在就引入 filtered-data 变量
- 不要同时再发散出更多结构分支

## 当前一句话策略

下一轮的核心不是“把 `aic_refiner` 做强”，而是：

> 把它有用的 recall / boundary 追回能力，压缩进 `keepbone_cutgap_v2` 的局部 keep cue 里，同时用更强的 precision veto 去修 `1433 / 1475` 这类 precision 崩坏样本。
