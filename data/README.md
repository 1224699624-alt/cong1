# 数据目录说明

- `raw/`：原始 TSRS 数据；不要改动结构。
- `raw_variants/`：隔离的 TSRS 过滤、重标注和对比度变体；每个变体有自己的 manifest/metadata。
- `remote_variants/RAM-W600/`：已同步并核对的 RAM-W600 images 与 train/val/test 多标签 masks。
- `yolo/`：旧 YOLO 格式派生数据；其生成脚本和历史记录保留。

TSRS 与 RAM 的实验、输出和报告必须继续分开。原始/私有医学数据默认不提交普通 Git blob；备份时使用私有仓库、Git LFS 或受控对象存储。
