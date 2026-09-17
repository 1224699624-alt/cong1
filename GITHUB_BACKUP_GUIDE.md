# GitHub 备份说明

本项目目前没有执行 `git init`、commit 或 push。清理工作只准备了本地目录和忽略规则。

## 默认会进入 Git 的内容

- `scripts/`、`configs/`、`run_*.sh`；
- `research-workflow/`、`research-wiki/`、`paper/`、`paper_zh/`、`patent/`；
- `outputs/analysis/` 中的 JSON/CSV/表格；
- 各实验目录中的 `result.json`、`protocol.json`、`history*`、`manifest*`、`README.md` 和日志；
- `outputs/artifact_bundles/r351_20260917/` 中的报告、结果和 provenance；
- 根目录 `MANIFEST.md`、`README.md`、`AGENTS.md` 及本说明。

## 默认只保留在本地

忽略规则不会删除这些文件，只防止它们被普通 `git add .` 意外提交：

- 原始/私有医学图像和标签：`data/raw/`、`data/raw_variants/`、`data/remote_variants/`；
- checkpoint：`checkpoints/`、`outputs/**/*.pth`、`outputs/**/*.pt`；
- 先验数组和 nnU-Net预处理：`outputs/**/*.npy`、`outputs/**/*.b2nd`、`outputs/**/*.pkl`；
- 预测掩膜、临时归档和运行缓存。

这些本地数据已经保留并核对。R351 的 RAM 原始数据与 R323 先验也可从 `outputs/artifact_bundles/r351_20260917/remote_archives/ARCHIVE_MANIFEST.json` 对照，但归档 tar 已清理，正式解包目录才是当前数据源。

## 建议的备份流程

1. 使用私有 GitHub 仓库；医学数据是否允许上传由项目所有者决定。
2. 先运行 `git status --ignored`，确认研究记录没有被忽略。
3. 小型结果文件先提交；大模型、先验数组和数据通过 Git LFS、私有对象存储或服务器快照保存。
4. 若使用 Git LFS，先针对明确的 checkpoint/数据模式执行 `git lfs track`，再提交 `.gitattributes`；不要强制添加整个 `outputs/`。
5. 提交前运行本地完整性检查：R351 归档中的 `RESUME.md`、`CURRENT_FILES_MANIFEST.json`、`provenance.json` 和 `CLEANUP_DELETION_MANIFEST_20260917.json` 必须存在。

## 当前重要入口

- 研究状态：[R351_TRACKER](research-workflow/refine-logs/R351_TRACKER.md)
- 清理记录：[cleanup_notes](cleanup_notes.md)
- R351恢复：[RESUME](outputs/artifact_bundles/r351_20260917/RESUME.md)
- R351代码快照：[CURRENT_FILES_MANIFEST](outputs/artifact_bundles/r351_20260917/CURRENT_FILES_MANIFEST.json)
- 删除审计：[CLEANUP_DELETION_MANIFEST](outputs/artifact_bundles/r351_20260917/CLEANUP_DELETION_MANIFEST_20260917.json)
