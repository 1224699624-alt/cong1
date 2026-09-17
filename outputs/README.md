# Outputs 目录说明

这里保存实验产物，不按“当前最好”删除历史方案。

- `analysis/`：指标 JSON/CSV、逐图评估和汇总表，论文写作优先读取这里。
- `priors/`：TSRS/RAM 先验图；每个变体对应不同实验假设，不能按文件大小合并。
- `nnunet/`：TSRS nnU-Net 数据、checkpoint、预测和训练产物。
- `ram_w600/`：RAM-W600 各版本模型、审计和官方指标。
- `artifact_bundles/`：服务器恢复包和带 provenance 的实验快照。
- `bridge_logs/`：实验启动和运行日志。
- `visualizations/`、`manual_review/`：论文图和人工复核证据。

可再生的 `.npy`、`.b2nd`、`.pkl`、`.pth`、`.pt` 和预测目录默认由根目录 `.gitignore` 排除，但本地文件仍保留。实验记录（JSON/CSV/MD/log/manifest）应提交到备份仓库。
