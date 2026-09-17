# 初始实验结果：R248 Closure → R251 Reverse-Background Channel

**日期**: 2026-07-10  
**计划**: `research-workflow/refine-logs/EXPERIMENT_PLAN.md`  
**范围**: `TSRS_RSNA-Epiphysis`；clean-test-v2 锁定；无 mask 写入

## M89：R248 Context-Crop Selector — NO-GO

| Run | Evidence | Key Result | Status |
| --- | --- | --- | --- |
| R248 | 96 original-val images, 8,568 OOF rows | 原始 risk grid 上限 0.50，低于全部 OOF risk probability（min 0.702877），所有配置为空 | INVALID_GRID |
| R248b | 同一 OOF rows 的 reachable regrid | 68/80 configs 非空，0 passing；best 4 images，2 safe / 2 risk，Recall -0.000015604 | DONE_NO_GO |

结论：局部 crop 双头评分器存在弱区分信号，但无法达到骨头保留所需精度。禁止进入 mask editor 和 clean-test-v2。

## M90：R251-S0 Reverse-Background Channel — SANITY PASS

训练目标：在 GT-free R244-style candidate corridor 内，用 `~bone_GT` 作为背景正类；保骨负类来自 corridor/ring/interior 的 bone GT。推理函数只接收 image + anchor。

| Metric | Value |
| --- | ---: |
| cases | 8/8 train cases |
| sampled pixels | 32,768 train / 65,536 eval |
| unique corridor background-positive pixels | 3,740 |
| ROC-AUC | 0.936614 |
| Average Precision | 0.918882 |
| probability range | 0.000744–0.998317 |
| gate | PASS |

解释：这证明反向背景通道在 train-only overfit 条件下可学习，不是 original-val 泛化证据。

## M90：R251 Candidate Selector Pipeline Smoke — ENGINEERING PASS

| Item | Result |
| --- | --- |
| train / val | 2 / 2 images |
| candidate rows | 21 |
| coverage | 2/2 complete, 0 skipped |
| calibration | score min/max/quantiles + safe/risk ROC-AUC/AP written |
| threshold grid | configured thresholds + reachable quantiles |
| promotion guard | `is_fullval=false`; passing config forced false |

2-case transfer safe-useful AUC was only 0.264706; this is intentionally underpowered and does not decide R251.

## M90：R251 Full Original-Val — RUNNING

- Remote screen: `r251_reverse_bg_selector_fullval`
- Train supervision: 32 usable original-train corridor cases
- Evaluation: all 96 original-val images, R244-consistent candidates
- Output isolation: `outputs/analysis/r251_reverse_background_candidate_selector_fullval*`
- Gate: >=10 selected images, >=8 safe-useful, <=2 risk, Recall >=0, cut GT foreground <=0.05, positive Boundary IoU/F1, negative gap FP, component-count MAE <=0
- clean-test-v2: not read
- masks: not written

## Current Bridge Status

- Must-run completed: R251-S0 sanity
- Must-run active: R251 full original-val selector audit
- Ready for auto-review-loop: NO
- Next decision: wait for full R251 JSON. Passing config → review before isolated R253 masks; no-go → R252 generator risk reduction.

