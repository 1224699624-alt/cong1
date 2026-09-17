# 项目进度总结（中文易读版）

**更新时间**: 2026-06-17

## 一、我们现在在做什么

当前项目目标很明确：

- 任务：儿童掌骨骨骺分割
- 主线：`YOLO+SAM` + error refiner
- 最终目标：在 `TSRS_RSNA-Epiphysis_clean_test_v2` 上追平或超过 `ARAA`

当前目标分数：

- `ARAA` clean-test-v2 Dice = `0.911766`

当前内部最强基线：

- `keepbone_cutgap_v2` clean-test-v2 Dice = `0.910029`

## 二、已经明确失败或不够好的线

### 1. `aic_refiner`

结论：

- recall 和 boundary 更高
- 但 precision 崩得太多
- 整体不能升级主线

### 2. `R011 = keepbone_cutgap_v3`

结果：

- 原始 test Dice = `0.892387`
- clean-test-v2 Dice = `0.909178`

结论：

- 低于当前内部最强主切片基线 `0.910029`
- 这条线失败

并且 review 也已经给出：

- `not ready`
- `back to implementation`

所以流程已经明确回到 bridge。

## 三、已经验证过但还不够的线

### `R010 = v2_precision_tune`

主切片：

- clean-test-v2 Dice = `0.910679`

结论：

- 是小幅正向
- 但 hard-case 仍不够干净

### `R012 = v2_precision_tune_thr054`

这是在 `R010` 基础上做的更保守阈值补实验：

- 不重新训练
- 直接复用 `R010` checkpoint
- 把固定阈值提到 `0.54`

结果：

- 原始 test Dice = `0.893033`
- clean-test-v2 Dice = `0.910030`

结论：

- 这条线几乎只是**打平**当前内部最强主切片基线
- 它说明更保守的阈值确实能提高 precision
- 但它还不足以算我们真正想要的突破

## 四、现在正在跑什么

### `R013 = allstack_anatomy_roi_boundary_prefgate_trimfirst_sam_hqsam`

这条线为什么值得继续试：

- 它不是我临时发明的新结构
- 而是当前代码里已经存在的现成分支
- 它本身就是偏 precision-safe、偏 boundary preference + trim 的方向
- 和我们现在“需要把 precision 控住”的目标是一致的

当前远端运行信息：

- 服务器：`10.1.115.157`
- 会话：`tmux:r013_prefgate`
- 日志：
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/bridge_logs/r013_prefgate.log`

## 五、现在为什么还没重新进入 `/auto-review-loop`

因为 ARIS 现在走得是正确流程：

1. `R011` 失败
2. review 把流程打回 implementation
3. `R012` 只做到打平，不足以结束 bridge
4. 所以当前还应该继续 `experiment-bridge`

只有当新的候选真正跑出足够强的结果，才应该重新进入 `/auto-review-loop`。

## 六、当前最该记住的一句话

`R011` 已失败，`R012` 只做到打平，所以当前 ARIS 仍在正常继续 `experiment-bridge`，已经发车新的桥接候选 `R013`。
