# 历史实验墓碑与归档索引 (History & Tombstones)

本目录记录了 2026-09-29 仓库清洗中建立的全部结构化索引、决策依据与被清理文件的去向清单。

## 目录索引

- **`EXPERIMENT_LEDGER.md`**：历史实验总账本。记录了历史上各阶段实验（R0xx 到 R351、RAM 细化研究、S/O 对照）的关键结论、数据集、指标变化及后继版本。
- **`SUPERSEDED_ROUTES.md`**：被否决/取代的技术路线总结。系统阐述了早期 YOLO+SAM、MaskDINO、复杂身份上下文、局部梯度监督等路线为何停止，防止后续重复试错。
- **`DELETION_MANIFEST.json`**：机器可读的删除产物清单。包含每个被清理文件从 main 下线的原因、替代版本及恢复标记。
- **`SCRIPT_USAGE_AUDIT.csv`**：`scripts/` 目录历史脚本的依赖与引用审计结果。
- **`EXACT_DUPLICATES.csv`**：全仓库完全相同文件的去重记录与规范主路径（canonical path）。

---

## 历史文件恢复指南

为了保证学术研究与工程复现的严格可追溯性，本次清洗**未修改 Git 提交历史，也未 force-push**。所有被移除的早期可视化图、日志、中间结果及旧版启动脚本均完好保存在清洗前创建的保护标签中：

- **清洗保护标签**：`pre-cleanup-2026-09-29`
- **基础提交点**：`118fe1f`

### 恢复方式示例

1. **临时切换到清洗前完整仓库**：
   ```bash
   git checkout pre-cleanup-2026-09-29
   ```

2. **按需恢复特定的历史文件或文件夹**（例如恢复某个历史脚本或输出图）：
   ```bash
   # 恢复某个被清理的旧脚本
   git checkout pre-cleanup-2026-09-29 -- scripts/train_r351_dual_prior.py

   # 恢复某个历史实验日志
   git checkout pre-cleanup-2026-09-29 -- outputs/bridge_logs/...
   ```

3. **查看某个历史文件的最后修改记录**：
   ```bash
   git log pre-cleanup-2026-09-29 -- <file-path>
   ```
