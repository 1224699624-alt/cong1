# 自动循环状态

**日期**: 2026-06-17

## 当前 ARIS 阶段

- `result-to-claim`: 已完成，用于判断 `aic_refiner` 这条线当前支持什么结论
- `research-refine`: 已完成，用于整理下一轮主候选修补方向
- `auto-review-loop`: 已经对 `R011` 跑过一轮外部 review，并明确把流程路由回 implementation
- `experiment-bridge`: 正在继续推进新的候选

## 已完成的关键结果

### `R010 = keepbone_cutgap_v2_precision_tune`

- clean-test-v2 Dice: `0.910679`
- 结论：主切片小幅提升，但 hard-case 仍不够干净

### `R011 = keepbone_cutgap_v3`

- 原始 test Dice: `0.892387`
- clean-test-v2 Dice: `0.909178`
- 结论：低于当前内部最强主切片基线 `0.910029`，失败

### `R012 = keepbone_cutgap_v2_precision_tune_thr054`

- 原始 test Dice: `0.893033`
- clean-test-v2 Dice: `0.910030`
- 结论：几乎只是打平当前内部最强主切片基线，不足以算明显正向突破

## 当前为什么还没重新进入 `/auto-review-loop`

原因很明确：

- `R011` 失败后，review 已经把流程路由回 implementation
- `R012` 虽然更保守，但也没有给出足够强的新结果
- 所以当前正确流程仍然是继续 `experiment-bridge`

## 当前正在跑的实验

### `R013 = allstack_anatomy_roi_boundary_prefgate_trimfirst_sam_hqsam`

这条线的定位是：

- 不再继续修 `R011 v3`
- 直接切到一个代码里已经存在、并且更偏 precision-safe 的现成分支
- 看它能否在原始 test 上至少给出更稳的 precision / boundary 组合

远端运行信息：

- 服务器：`10.1.115.157`
- 目录：`/home/shenzeyu/workspace/YOLO_SAM_generic_src`
- 会话：`tmux:r013_prefgate`
- 日志：
  - `/home/shenzeyu/workspace/YOLO_SAM_generic_src/outputs/bridge_logs/r013_prefgate.log`

## 当前一句话总结

`R011` 已失败并被 review 路由回 implementation；`R012` 只做到几乎打平、不足以结束 bridge；当前 ARIS 正在正常继续往下走，已经发车新的桥接候选 `R013`。
