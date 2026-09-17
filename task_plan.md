# Task Plan: YOLO+SAM 项目保守清理与 GitHub 备份准备

## Goal
在不删除实验记录、原始数据、可复现实验脚本、重要 checkpoint 和评估证据的前提下，清理明确的缓存、重复中间产物和无价值临时文件，并生成可审计的清理报告。

## Phases
- [x] Phase 1: 建立保护清单并盘点本地/远程资产
- [x] Phase 2: 分类候选清理项，核对是否被实验记录或脚本引用
- [x] Phase 3: 先创建可恢复的清理归档，再执行最小删除
- [x] Phase 4: 运行文件完整性检查，生成 GitHub 备份说明

## Protection Rules
- 不删除 `research-workflow/`、`research-wiki/`、`scripts/`、`configs/`、`run_*.sh` 和实验日志。
- 不修改或删除 `data/raw/`、`data/raw_variants/`；TSRS 与 RAM 数据保持分开。
- 不删除有 `result.json`、`protocol.json`、`manifest`、`checkpoint` 或实验记录引用的输出目录。
- 不删除仍可能用于恢复训练的最佳/最终 checkpoint；中间 checkpoint 只有在确认有最终替代和哈希记录后才可归档。
- 首轮只清理明确的缓存、临时目录、重复归档和无引用的中间预测；任何不确定项保留。
- 不向 GitHub 推送，先生成本地可审查的备份结构和报告。

## Status
**Complete** - 清理、完整性核对、GitHub忽略规则和备份说明均已完成；没有执行 GitHub 推送。

## Errors Encountered
- 项目当前没有 `.git` 目录；已生成 GitHub 备份说明，但本次不执行推送。
