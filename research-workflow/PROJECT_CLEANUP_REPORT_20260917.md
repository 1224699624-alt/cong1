# YOLO+SAM 项目清理报告（2026-09-17）

## 范围

本次目标是为后续 GitHub 私有备份整理本地项目，同时保留继续实验和论文写作所需证据。没有执行 `git init`、commit、push，也没有登录或修改 GitHub。

## 已保留

- 全部 `research-workflow/`、`research-wiki/`、`scripts/`、`configs/`、`run_*.sh`；
- 全部 TSRS/RAM 原始数据和变体，且两个数据集目录未合并；
- `outputs/analysis/` 所有指标 JSON/CSV；
- 全部 `outputs/priors/`、`outputs/nnunet/`、`outputs/ram_w600/` 和历史 artifact bundles；
- R351 代码、结果、checkpoint、日志、provenance 和恢复说明；
- 论文、专利、历史方案与失败实验记录；
- `checkpoints/`、`runs/`、`.tmp/` 中未能证明无用的内容。

## 已删除

仅删除两类可验证对象，详细路径、大小和 SHA256 见：
`outputs/artifact_bundles/r351_20260917/CLEANUP_DELETION_MANIFEST_20260917.json`。

1. R351 RAM 数据归档的五个重复 tar：数据已经解包到正式目录，并在删除前核对远程/本地文件数及字节数一致；删除约 4.12 GB重复归档。
2. 各级 `__pycache__` 中的 723 个 `.pyc` 文件：可由 Python 重新生成；删除约 7.8 MB缓存。

没有删除任何实验记录或结果文件。R351 数据归档 tar 的原始文件名和 SHA256仍保留在 `remote_archives/ARCHIVE_MANIFEST.json`，正式数据位于 `data/remote_variants/RAM-W600/` 和 `outputs/priors/r323_ram_native_r317_single_seed/`。

## 完整性核对

- RAM images：618 个，392,779,020 bytes；
- RAM train masks：425 个，2,379,749,200 bytes；
- RAM val masks：69 个，376,009,632 bytes；
- RAM test masks：124 个，676,087,072 bytes；
- R323 prior：989 个，293,662,368 bytes；
- 删除后 `pyc_remaining=0`，R351重复 tar数量为0；
- R351 shared tests 6/6、evaluation tests 11/11、self-test 和 `py_compile` 均通过。

## GitHub备份策略

`.gitignore` 已从“忽略整个 outputs”调整为“保留实验记录、忽略可再生大二进制和私有医学数据”。具体提交策略见根目录 `GITHUB_BACKUP_GUIDE.md`。这样普通 `git add .` 不会漏掉研究记录，也不会误把数十 GB 数据和 checkpoint 当作普通 Git blob。
