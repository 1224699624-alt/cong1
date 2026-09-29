# GitHub 同步与仓库维护说明 (2026-09-29)

本项目已建立正式 GitHub 远端仓库并处于持续维护中。
远端地址：`https://github.com/1224699624-alt/cong1.git`

---

## 1. 保护点与版本追溯机制

为保证医学研究的可追溯性与完整性，本项目遵循**非破坏性清洗**原则：
- **历史完整保护标签**：`pre-cleanup-2026-09-29`
  包含 2026-09-29 清洗前全部 3,589 个文件、历史逐图可视化面板、所有 daemon 日志及中间过渡脚本。
- **清理工作分支**：`repo-cleanup-2026-09`
  用于收敛工作树体积、整理有效启动器并生成实验台账。
- **严禁强制推送**：不重写 Git 历史（不执行 filter-repo、BFG 或 force-push）。

若需查看或还原清洗前的历史产物，可直接通过保护标签检出：
```bash
git checkout pre-cleanup-2026-09-29 -- <path/to/historical/file>
```

---

## 2. 仓库内容跟踪与过滤策略

### 纳入 Git 跟踪的内容
- 核心代码：`scripts/`（经审计保留的 436 个当前/通用模块）；
- 有效入口：`launchers/ram/` 与 `launchers/tsrs/`；
- 研究文档：`research-wiki/`、`research-workflow/`、`codex_handoff/`、`paper/`、`patent/`；
- 结构化历史：`docs/history/`（包含实验总账本、路线复盘、删除清单与去重表）；
- 核心指标数据：`outputs/analysis/` 中的核心评估 JSON / CSV，以及 09-22~09-29 关键结论文档。

### 默认排除在 Git 之外的内容（仅保留本地/服务器）
- 原始与变体医学图像：`data/raw/`、`data/raw_variants/`、`data/remote_variants/`；
- 大规模模型权重与中间 Checkpoint：`checkpoints/`、`*.pth`、`*.pt`；
- 预处理与先验大数组：`*.npy`、`*.b2nd`、`*.pkl`；
- 调试临时缓存、`__pycache__` 及大批量运行日志。

---

## 3. 标准提交流程

1. 检查工作区状态：
   ```bash
   git status
   ```
2. 确认未添加未受控的大型二进制产物：
   ```bash
   git diff --stat
   ```
3. 提交与推送至清理分支：
   ```bash
   git add <modified-files>
   git commit -m "..."
   git push origin repo-cleanup-2026-09
   ```
4. 通过 GitHub PR 审查并合并至 `main`。

---

## 4. 当前核心入口

- **当前研究现状与结论**：[`research-wiki/experiments/ram_so_heatmap_20260929.md`](research-wiki/experiments/ram_so_heatmap_20260929.md)
- **Agent 交接规格与边界**：[`codex_handoff/RAM_SO_HEATMAP_CONDITIONAL_HANDOFF/00_README_AND_START.md`](codex_handoff/RAM_SO_HEATMAP_CONDITIONAL_HANDOFF/00_README_AND_START.md)
- **历史实验总账本**：[`docs/history/EXPERIMENT_LEDGER.md`](docs/history/EXPERIMENT_LEDGER.md)
- **被取代路线深度复盘**：[`docs/history/SUPERSEDED_ROUTES.md`](docs/history/SUPERSEDED_ROUTES.md)
- **删除产物检索清单**：[`docs/history/DELETION_MANIFEST.json`](docs/history/DELETION_MANIFEST.json)
