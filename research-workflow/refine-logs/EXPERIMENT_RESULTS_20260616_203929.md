# 实验结果总览

## 目标基线

### 外部目标：ARAA

- 主切片：`TSRS_RSNA-Epiphysis_clean_test_v2`
- Dice: `0.911766`
- IoU: `0.838436`
- Precision: `0.910177`
- Recall: `0.914691`
- Boundary IoU: `0.235265`

### 当前内部最强：`keepbone_cutgap_v2`

- clean-test-v2 Dice: `0.910029`
- clean-test-v2 Boundary IoU: `0.228489`
- 原始 test Dice: `0.893694`

## `aic_refiner` 的结论

- clean-test-v2 Dice: `0.905869`
- clean-test-v2 Precision: `0.896434`
- clean-test-v2 Recall: `0.916901`
- clean-test-v2 Boundary IoU: `0.238092`

结论：

- recall 和 boundary 更高
- 但 precision 掉太多
- 整体 Dice 不够，所以不能升级主线

## `R010 precision_tune`

- clean-test-v2 Dice: `0.910679`
- clean-test-v2 Precision: `0.912659`
- clean-test-v2 Recall: `0.909911`
- clean-test-v2 Boundary IoU: `0.232611`

结论：

- 主切片小幅正向
- 但 hard-case 仍然不够干净

## `R011 v3`

### 原始 test

- Dice: `0.892387`
- IoU: `0.808634`
- Precision: `0.898868`
- Recall: `0.889522`
- Boundary IoU: `0.215672`

### clean-test-v2

- Dice: `0.909178`
- IoU: `0.834020`
- Precision: `0.912985`
- Recall: `0.906651`
- Boundary IoU: `0.229008`

### 对比当前内部最强主切片基线

- 基线 Dice: `0.910029`
- `R011` Dice: `0.909178`
- 差值: `-0.000851`

### 结论

- `R011` 已完整跑完
- 主切片仍低于当前内部最强基线
- 因此这条线目前也不够，不能进入 `/auto-review-loop`

## 当前阶段判断

- 还在 `experiment-bridge`
- 只有当某条候选完整跑完，并且主切片与 hard-case 都足够强，才会进入 `/auto-review-loop`
