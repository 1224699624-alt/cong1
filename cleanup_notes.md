# Cleanup Notes

本文件记录项目清理期间的证据、保护判断和保留/删除决定。删除动作必须先在 `task_plan.md` 中明确范围，并在最终报告中记录原路径、原因、大小和可恢复归档位置。

## 盘点结论（2026-09-17）

- 本地约 60 GB：`outputs` 约52.6 GB，`data` 约5.94 GB，`.tmp`约1.77 GB。
- 保护：所有研究记录、代码/launcher、原始数据与变体、先验、checkpoint、评估结果、历史 artifact bundle、论文/专利文档。
- 明确可清理候选：`outputs/artifact_bundles/r351_20260917/remote_archives/*.tar`（数据已经解包且远端/本地文件数和字节数核对一致，属于重复归档）；各级 `__pycache__/` 和 `*.pyc`（可重建缓存）。
- 暂不清理：`outputs/priors` 三个大型关系图目录、`outputs/nnunet` 各实验目录、`.tmp/external`、`outputs/deploy_cache`、`runs/detect`、`checkpoints`。它们仍可能支持复现实验或后续迁移，且缺少足够证据证明无用。
- 研究记录与失败方案不按“当前最好”筛选，全部保留。

## 首轮删除清单（待执行）

- `outputs/artifact_bundles/r351_20260917/remote_archives/r351_ram_*.tar`：5个归档，共约3.84 GB。对应RAM数据和先验已解包到正式目录，并已核对远端/本地计数与字节数；保留 `ARCHIVE_MANIFEST.json`，不删除解包后的数据。
- 所有路径末端名称严格为 `__pycache__` 的目录及其中 `.pyc`：723个文件、约7.8 MB。均可由Python重新生成，不含实验结果。
- 其他目录和文件本轮不删：尤其是 `outputs/priors`、`outputs/nnunet`、`outputs/ram_w600`、`outputs/analysis`、`outputs/artifact_bundles` 的非R351归档、`.tmp`、`checkpoints`、`runs` 和全部研究记录。
