# v2 主线切换记录

旧版 `FINAL_PRIOR_PILOT_REPORT.md` 与旧审计结果保留为历史记录。v2 将 R351 cross-dataset dual-prior 降级为 optional extension。

核心入口现在是：

- TSRS `T0` / `tsrs_native_continue`：只加载 TSRS native backbone。
- TSRS `T1` / `tsrs_seam_prior`：只加载 TSRS native backbone 和 TSRS seam branch；不加载 RAM overlap checkpoint。
- RAM `R0` / `ram_native_continue`：只加载 RAM native checkpoint。
- RAM `R1` / `ram_overlap_completion`：只加载 RAM native checkpoint、RAM overlap prior 和 RAM-native completion refiner；不加载 TSRS prior。

旧 `A/B/C/D` 分支、R351 cross-transfer 权重和 dual runner 代码不删除，保留在 optional/extension 语义下，不得作为 core final model 或核心论文表格入口。

对应短 pilot 命令：

```bash
bash run_final_prior_tsrs.sh T0
bash run_final_prior_tsrs.sh T1
bash run_final_prior_ram.sh R0
bash run_final_prior_ram.sh R1
```

这些命令仍只用于短 pilot，输出必须使用新的隔离目录；现有旧 pilot 目录不覆盖。full training、multi-seed、额外 backbone 和 test 仍按 v2 顺序等待 core pilot 判断。
