# R251-S0 实验代码审阅

**日期**: 2026-07-10  
**审阅模式**: 独立 Codex reviewer，两轮审阅  
**结论**: `APPROVED_FOR_REMOTE_SMOKE`

## 审阅文件

- `scripts/train_r251_reverse_background_channel.py`
- `run_r251_reverse_background_channel_smoke.sh`
- 计划锚点：`research-workflow/refine-logs/EXPERIMENT_PLAN.md` 的 R251-S0

## 首轮阻断项

1. 初版监督覆盖整个 `anchor & ~GT`，可能只学到普通边界过分割，未限定到候选骨缝 corridor。
2. sanity gate 失败仍返回 0 并保存模型，无法阻止后续 full run。
3. dataset/split 只在 launcher 中声明，没有在 Python 内锁死；`clean_test_v2_used=false` 可能被错误调用伪造。
4. 缺 anchor 或单类样本会被跳过，少量样本也可能通过 gate。

## 已实施修复

- 使用与 R244 同源的 GT-free `candidate_components(anchor, image, ...)` 构建 corridor；候选资格不读取 GT。
- 正类改为 `corridor & ~bone_GT`；保骨负类包含 `corridor & bone_GT`、corridor ring 和 anchor interior bone pixels。
- 推理函数仍为 `predict_background_prob(model, image, anchor)`，特征只来自 image/anchor。
- Python 内强制：仅 `TSRS_RSNA-Epiphysis`、原始 `data/raw`、`train -> train` overfit smoke；拒绝 clean-test-v2、Articular-Surface 和其他 split/root。
- gate 要求严格相同且完整的 8 个 train case，缺失/跳过/身份不一致均失败。
- AP gate 从 0.25 提高到 0.75；失败时写诊断 JSON、返回 exit 2、不保存模型。
- launcher 通过 `${PIPESTATUS[0]}` 保留 Python 失败码。
- JSON/model/log 使用时间戳文件并更新 fixed latest alias；记录随机种子、版本、case names 和区域像素计数。

## 复审结论

复审确认四个阻断项均已关闭，没有发现剩余部署阻断问题：

- 方法与 R251-S0 反向背景监督计划一致；
- 未发现推理时 GT 泄漏；
- 评估标签来自数据集 GT；
- 范围、完整性、失败传播和输出隔离满足 sanity-first 要求；
- 本地 AST/import/`--help`/py_compile 检查通过。

## 非阻断后续加固

- 消费模型时同时校验当前 run exit code 与 JSON 时间戳，避免误用旧 fixed model alias。
- full R251 前补充更多 argparse 边界校验，并明确 `max_candidates_generated` 是每个 distance percentile 的上限。
- 若 8 个 usable case 不足，在更早异常路径也落盘 selection audit，便于远程诊断。

## Full R251 Candidate Selector Review

**审阅结论**: `APPROVED_FOR_LONG_REMOTE_RUN`

首轮 full-script 阻断项及修复：

1. 增加候选分数 min/max/quantiles、safe-useful ROC-AUC/AP、risk ROC-AUC/AP，并在阈值网格之前计算；网格自动加入观测 q05/q10/q25/q50/q75/q90/q95，避免重复 R248 unreachable-grid 问题。
2. 记录 expected/processed/zero-candidate/skipped image names；只有 `limit=0` 且所有 original-val image 完整处理时才允许 `passes_gate=true`，pilot 永远不能 promotion。
3. 修正 `choose_best` 的零值处理，理想的 `mean_cut_gt_fg_frac=0` 不再被当作缺失。
4. 强制 `max_actions_per_image=1`，保证单 cut metric delta 的有效性。

复审确认：train GT 只生成训练监督；val candidate eligibility、概率打分和 top-1 selection 均为 GT-free；val GT 只用于允许的 original-val audit 和阈值选择。B1 八项 gate 与计划完全一致。

