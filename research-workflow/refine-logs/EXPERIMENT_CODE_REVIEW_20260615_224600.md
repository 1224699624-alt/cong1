# 实验代码检查

**日期**: 2026-06-15 22:46  
**模式**: `[local-only]`

说明：
- 这轮没有额外调用独立 reviewer 做代码检查
- 所以这里是执行器本地的发车前检查记录

## 本轮检查的文件

- `scripts/train_error_refiner.py`
- `scripts/infer_error_refiner.py`
- `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune.sh`
- `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v3.sh`

## 检查结论

### 1. 方案和修补计划是否一致

一致。

- `v2_precision_tune`：低风险版本，只调 loss / threshold / precision 相关权重
- `v3`：结构修补版，在 `keepbone_cutgap_v2` 主体上加入：
  - 局部 AIC 风格 keep cue
  - 额外 precision veto

这和 [NEXT_CANDIDATE_REPAIR_PLAN.md](/g:/gutou/YOLO+SAM/research-workflow/refine-logs/NEXT_CANDIDATE_REPAIR_PLAN.md) 的设计一致。

### 2. 参数接口是否暴露完整

是。

- `v2_precision_tune` 只复用现有 argparse 参数
- `v3` 新增的是新的 `target-mode`
- 没有偷偷写一堆外部不可控的 launcher 参数

### 3. 路由与模式识别是否补齐

已补齐。

- `keepbone_cutgap_refiner_v3` 已加入：
  - parser choices
  - anatomy mode 路由
  - target 生成路径
  - inference allowlist

### 4. 结果与输出是否可隔离复现

是。

新 launcher 都有独立：

- checkpoint 名
- threshold sweep 文件名
- 输出 experiment 目录名

不会和旧 run 混在一起。

### 5. 立即阻塞问题

无。

本地语法检查已通过：

```bash
python -m py_compile scripts/train_error_refiner.py scripts/infer_error_refiner.py
```

## 非阻塞风险

### `v3` 风险

- `v3` 直接改了 stage2 correction path
- 因此最终表现必须靠 `1433 / 1475 / 2982 / 3040` 这些 hard-case 来判

### `precision_veto` 风险

- 如果 trim 太强，可能会把 `3040` 这类原本有益的恢复一起削掉
- 这也是为什么先跑了 `v2_precision_tune`，再上 `v3`

## 发车决策

- 先发车：`v2_precision_tune`
- 如果 low-risk 版本不够，就继续发车：`v3`
