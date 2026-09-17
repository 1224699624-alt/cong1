# `/experiment-bridge` 执行化输入稿

**日期**: 2026-06-15 20:59  
**来源**:
- `research-workflow/RESEARCH_BRIEF.md`
- `research-workflow/refine-logs/EXPERIMENT_PLAN.md`

## 任务目标

请基于当前 `YOLO+SAM` 仓库，执行一轮面向 `TSRS_RSNA-Epiphysis` 的桥接实验，把现有 plan 落成可运行实验并收集首轮结果。

本轮不是开放式找新 idea，而是沿着当前已明确的主线推进：

- 主目标：在 `TSRS_RSNA-Epiphysis_clean_test_v2` 上逼近或超过 ARAA `Dice = 0.911766`
- 次目标：在原始 `TSRS_RSNA-Epiphysis` test 上不出现明显退化
- 关键瓶颈：薄骨骺保留、相邻骨缝保持、边界完整性

## 本轮执行假设

- 远端主工作区：`/home/shenzeyu/workspace/YOLO_SAM_generic_src`
- 环境：`conda activate yolo-sam-gpu`
- 后端：`ssh`
- 优先复用现有脚本，不额外发明新训练框架
- 本轮主候选默认使用：`allstack_anatomy_roi_aic_refiner_sam_hqsam`
- 当前最强内部锚点：`allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`
- clean-test-v2 仍作为主决策切片，原始 test 作为控制切片

## 必须复用的现有脚本

- 训练主候选：
  - `run_epiphysis_allstack_anatomy_roi_aic_refiner.sh`
- 当前最强内部锚点训练脚本：
  - `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v2.sh`
- clean-test-v2 评测脚本：
  - `run_epiphysis_clean_test_keepbone_cutgap_refiner_v2.sh`
  - 注意：该脚本可通过设置 `REFINER_EXP=...` 复用于别的 refiner，不必新写 clean-test 脚本

## 锁定基线

### 外部基线

- 方法：ARAA
- clean-test-v2 最优 checkpoint：`98.pth`
- 目标分数：`Dice = 0.911766`
- 远端结果路径：
  - `/home/shenzeyu/workspace/ARAA-Net/outputs/tsrs_epiphysis_clean_test_v2_ckpt_sweep/epoch_98/metrics.json`

### 内部基线

- 方法：`allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`
- clean-test-v2 指标：
  - Dice `0.910029`
  - Boundary IoU `0.228489`
- 原始 test 对应输出应作为控制切片一并记录

## 本轮 must-run 里程碑

### M0: 基线冻结

目标：
- 把 ARAA 与当前最强内部 refiner 的对照表固定下来
- 确认后续所有比较都使用同一套指标口径

动作：
- 读取 ARAA clean-test-v2 指标
- 读取 `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam` 的原始 test 与 clean-test-v2 指标
- 将结果写入 `refine-logs/EXPERIMENT_RESULTS.md` 的基线段

通过门槛：
- 基线数值来源无歧义
- clean-test-v2 与原始 test 都能落到明确路径

### M1: 主候选完整跑通

目标：
- 用现有 launcher 完整运行 `allstack_anatomy_roi_aic_refiner_sam_hqsam`
- 获得原始 test 与 clean-test-v2 的成套指标

动作：
1. 运行训练与 threshold sweep：

```bash
bash run_epiphysis_allstack_anatomy_roi_aic_refiner.sh
```

2. 基于已有 test masks 复用 clean-test-v2 评测：

```bash
REFINER_EXP=allstack_anatomy_roi_aic_refiner_sam_hqsam bash run_epiphysis_clean_test_keepbone_cutgap_refiner_v2.sh
```

需要收集的产物：
- checkpoint
- threshold sweep json
- 原始 test `metrics.json`
- clean-test-v2 `metrics.json`
- clean-test-v2 summary markdown

关键路径预期：
- 原始 test 输出目录：`outputs/ablations/allstack_anatomy_roi_aic_refiner_sam_hqsam/TSRS_RSNA-Epiphysis/test/`
- clean-test-v2 输出目录：`outputs/ablations_variants/allstack_anatomy_roi_aic_refiner_sam_hqsam/TSRS_RSNA-Epiphysis_clean_test_v2/test/`

通过门槛：
- 训练、推理、评测全部成功
- 所有结果保存为可解析 json / md

### M2: 主候选对比判定

目标：
- 判断 `allstack_anatomy_roi_aic_refiner_sam_hqsam` 是否值得继续提升或推广

主判定规则：
- 优先看 clean-test-v2
- 再看原始 test 是否崩塌
- 再看难例表现是否与指标结论一致

推进条件：
- clean-test-v2 Dice `>= 0.910029` 且无明显 hard-case 恶化

强推进条件：
- clean-test-v2 Dice `>= 0.911766`

停止条件：
- clean-test-v2 Dice 明显低于 `0.910029`
- 或虽然指标接近，但 `2982 / 1475 / 3040 / 1433` 出现更严重骨缝闭合或薄骨骺丢失

### M3: 难例验证

目标：
- 检查增益是否真的来自“薄骨骺保留 + 骨缝保持”

必须检查的样本：
- `2982`
- `1475`
- `3040`
- `1433`

至少需要导出的内容：
- baseline / current best / main candidate 三方对比图
- 简短文字判读

判定重点：
- 是否减少薄骨骺漏分
- 是否避免骨间隙被错误连上
- 是否只是轮廓更平滑但解剖真实性更差

### M4: 过滤数据控制实验

仅在 M2 结果为“正向或接近正向”时再做。

目标：
- 判断可能的增益来自结构设计，还是主要来自 train/val 过滤

优先控制版本：
- `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_trainval_filtered_v1.sh`
- `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_badlabel_filtered_v1.sh`

规则：
- 必须保持独立输出目录
- 过滤数据结果绝不能和默认数据主线混报

## 本轮建议 run order

1. M0 基线冻结
2. M1 主候选 `allstack_anatomy_roi_aic_refiner_sam_hqsam`
3. M2 指标判定
4. M3 难例验证
5. 只有在 M2 正向时才进入 M4

## 本轮不做的事

- 不新造一个尚未在仓库落地的 refiner 架构
- 不同时并行很多新候选，先把 `aic_refiner` 这条历史主线验证清楚
- 不把 filtered-data 的收益直接算作架构收益

## 需要 bridge 额外补的最小实现项

如果 `experiment-bridge` 发现缺失项，只允许补最小缺口：

- 缺少统一结果汇总时，补一个 `refine-logs/EXPERIMENT_RESULTS.md`
- 缺少 hard-case 可视化汇总时，补一个轻量脚本或复用现有可视化逻辑
- 不要重写训练框架
- 不要改动原始数据结构

## 输出文件要求

本轮至少产出：

- `refine-logs/EXPERIMENT_RESULTS.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- 如有代码变更，记录到 `refine-logs/EXPERIMENT_CODE_REVIEW.md`

结果汇总至少要包含：

- ARAA clean-test-v2 对照
- current best internal refiner 对照
- `allstack_anatomy_roi_aic_refiner_sam_hqsam` 的原始 test / clean-test-v2 指标
- 对 `2982/1475/3040/1433` 的难例判断
- 是否进入 filtered control 的结论

## 最终一句话决策格式

桥接完成后，请明确给出以下三类之一：

- `PROMOTE`: 主候选已达到主线升级条件
- `ITERATE`: 主候选接近可用，但需要定向修补
- `REJECT`: 主候选没有优于当前内部最强线

## 可直接传给 `/experiment-bridge` 的一句话摘要

基于 `research-workflow/RESEARCH_BRIEF.md` 和 `research-workflow/refine-logs/EXPERIMENT_PLAN.md`，请在远端 `/home/shenzeyu/workspace/YOLO_SAM_generic_src` 上复用现有 launcher，优先执行 `allstack_anatomy_roi_aic_refiner_sam_hqsam` 这一主候选，先冻结 ARAA 与 `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam` 基线，再完成原始 test 与 `clean-test-v2` 双切片评测，并用 `2982/1475/3040/1433` 做难例解剖学验证，只有在结果正向时才进入 filtered-data 控制实验。
