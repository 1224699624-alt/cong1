# Task Plan (Historical Archive — 2026-09-17)

> **归档说明**：本计划文件为 2026-09-17 Git 仓库建立前的初版保守清理记录，其前提假设（“项目未执行 git init / 无 .git”）现已失效。
> **当前权威清理规范与执行记录请参见**：
> - GitHub Issue #1
> - [`docs/history/EXPERIMENT_LEDGER.md`](docs/history/EXPERIMENT_LEDGER.md)
> - [`docs/history/DELETION_MANIFEST.json`](docs/history/DELETION_MANIFEST.json)
> - [`docs/history/SUPERSEDED_ROUTES.md`](docs/history/SUPERSEDED_ROUTES.md)

---

## Historical Goal (2026-09-17)
在不删除实验记录、原始数据、可复现实验脚本、重要 checkpoint 和评估证据的前提下，清理明确的缓存、重复中间产物和无价值临时文件，并生成可审计的清理报告。

## Historical Phases
- [x] Phase 1: 建立保护清单并盘点本地/远程资产
- [x] Phase 2: 分类候选清理项，核对是否被实验记录或脚本引用
- [x] Phase 3: 先创建可恢复的清理归档，再执行最小删除
- [x] Phase 4: 运行文件完整性检查，生成 GitHub 备份说明

## Protection Rules (Historical)
- 不删除 `research-workflow/`、`research-wiki/`、`scripts/`、`configs/`、`run_*.sh` 和实验日志。
- 不修改或删除 `data/raw/`、`data/raw_variants/`；TSRS 与 RAM 数据保持分开。
- 不删除有 `result.json`、`protocol.json`、`manifest`、`checkpoint` 或实验记录引用的输出目录。
- 不删除仍可能用于恢复训练的最佳/最终 checkpoint；中间 checkpoint 只有在确认有最终替代和哈希记录后才可归档。
- 首轮只清理明确的缓存、临时目录、重复归档和无引用的中间预测；任何不确定项保留。
- 不向 GitHub 推送，先生成本地可审查的备份结构和报告。

## Status
**Historical Complete (2026-09-17)** - 现已由 2026-09-29 Issue #1 仓库清洗行动全面覆盖与更新。
