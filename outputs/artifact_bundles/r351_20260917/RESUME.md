# R351恢复说明

归档日期：2026-09-17。服务器 `/root/autodl-tmp/YOLO_SAM_generic_src` 当前无 R351进程；RAM pilot、TSRS pilot 和正向保护诊断均已结束。

## 已完成

- RAM overlap prior：6 epoch，425 train / 69 val，best epoch 6，未使用 test。
- TSRS dual-prior pilot：2 epoch，96 original-val R201，checkpoint已绑定并保存；从成熟 R350 初始化，native仅作独立gate。
- RAM dual-prior pilot：2 epoch，best epoch 1，69 validation，native DSC/IoU和5% overlap/pair MSD gate通过；相对R332 anchor有部分NSD/整体指标退化，不能称全面提升。
- TSRS positive-seam-guard：固定已有pilot checkpoint的post-hoc validation诊断，未重新训练，未使用 test。

## 归档内容

- `ram_pilot/`：RAM最终 result、history、manifest、best/last checkpoint、协议与身份审计。
- `tsrs_pilot/`：TSRS R201、结果、adapter权重、逐图比较和像素变化。
- `tsrs_positive_guard/`：固定推理策略诊断结果。
- `overlap/full/`：RAM真实多标签 overlap prior 权重、6轮历史和结果。
- `remote_archives/`：RAM完整图像、train/val/test masks、R323先验 tar归档；解包目标为项目根目录，`ARCHIVE_MANIFEST.json`记录归档哈希。
- `current_files/`：当前R351代码、启动脚本、计划、审查与骨干契约；`CURRENT_FILES_MANIFEST.json`记录哈希。
- `provenance.json`：主要源码和冻结权重哈希。

## 额度恢复后的第一步

先读取本文件和 `research-workflow/refine-logs/R351_TRACKER.md`，不要直接打开 test 做选择。最有价值的后续是：从原生 R202/R325 分别启动同一双先验结构的 matched native-start pilot，并做无先验、仅输入、仅输出、双先验四项消融。TSRS现有索引PNG无法表达真实投影重叠，因此其 overlap loss 仍应标记 unknown，不能伪造GT。
