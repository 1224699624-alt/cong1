---
type: paper
node_id: paper:yang2025_ramw600_multitask_wrist
title: "RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis"
authors: ["Songxiao Yang", "Haolin Wang", "Yao Fu", "Ye Tian", "Tamotsu Kamishima", "Masayuki Ikebe", "Yafei Ou", "Masatoshi Okutomi"]
year: 2025
venue: "NeurIPS 2025 Datasets and Benchmarks"
external_ids:
  arxiv: "2507.05193"
  doi: "10.52202/085713-3375"
  s2: null
tags: ["dataset", "wrist-xray", "instance-segmentation", "joint-space"]
added: 2026-07-22T04:21:23Z
---

# RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis

## One-line thesis
公开腕部 X 光多任务数据集和分割基准；即使整体骨分割 Dice 很高，重叠骨对边界仍是主要难点。

## Problem / Gap
腕骨在投影 X 光中互相遮挡，RA 骨侵蚀又改变轮廓，精确实例标注与边界预测都困难。

## Method
提供 618 张有 14 成员实例分割标注的腕部 BMP，以及骨侵蚀评分任务。分割 benchmark 在 512×512、患者 ID 分割的 train/valid/test 上训练 13 个监督模型（5 seeds），另评估 SAM/MedSAM 的单次推理。

## Key Results
主版本 arXiv v3 Table 4：SwinUMamba test All DSC 97.75%、NSD@2px 90.71%；普通 UNet 为 96.79%、83.36%；UNet++ 为 97.33%、86.99%；TransUNet 为 97.62%、89.48%。Table 12/13 的 14 个重叠骨对逐对 DSC/NSD 明显更低；例如 Cap–Tz 的 UNet DSC 49.31%、SwinUMamba 60.01%。论文没有测试 nnU-Net 或外部 O/seam 先验。完整核对见 [原论文证据](../../outputs/ram_original_paper_audit_20260927/RAM_W600_ORIGINAL_PAPER_EVIDENCE.md)。

## Assumptions
总体指标来自论文 test 集，512×512 输入、100 epochs、5 个监督模型随机种子；不能与本项目原生分辨率 val69 单次结果直接相减。

## Limitations / Failure Modes
不同骨对的重叠边界难度差异大。作者将 pisiform 和 triquetrum 合并为一类，反映单视角 X 光下的临床可辨性限制。模型家族对比不是某个注意力/skip 模块的隔离因果实验。

## Reusable Ingredients
高分辨率多尺度 skip、较远距离的骨形状上下文和边界保留是值得在现有 nnU-Net 中做单因素验证的方向；不宜直接因论文排名更换整套骨架。

## Open Questions
当前 nnU-Net 的高分辨率冻结特征，能否在固定 H0 第二成员漏检与匹配负成员间提供可分信息？现有 O 提前进入 encoder 是否产生真实增量，仍未受控验证。

## Claims
论文支持“复杂骨重叠仍是通用分割模型的难点”，不支持“输入端先验一定更好”或“当前方法优于论文 test benchmark”。

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
RAM 是当前主数据集。先保留成熟 nnU-Net/R332，借鉴 UNet++ 的边界/skip 融合或 SwinUMamba/TransUNet 的全局上下文思路，但必须在本地同起点、同预算、同评价协议下验证 H0 恢复和 FP/NSD 代价。

## Abstract (original)

> Rheumatoid arthritis (RA) is a common autoimmune disease that has been the focus of research in computer-aided diagnosis (CAD) and disease monitoring. In clinical settings, conventional radiography (CR) is widely used for the screening and evaluation of RA due to its low cost and accessibility. The wrist is a critical region for the diagnosis of RA. However, CAD research in this area remains limited, primarily due to the challenges in acquiring high-quality instance-level annotations. (i) The wrist comprises numerous small bones with narrow joint spaces, complex structures, and frequent overlaps, requiring detailed anatomical knowledge for accurate annotation. (ii) Disease progression in RA often leads to osteophyte, bone erosion (BE), and even bony ankylosis, which alter bone morphology and increase annotation difficulty, necessitating expertise in rheumatology. This work presents a multi-task dataset for wrist bone in CR, including two tasks: (i) wrist bone instance segmentation and (ii) Sharp/van der Heijde (SvdH) BE scoring, which is the first public resource for wrist bone instance segmentation. This dataset comprises 1048 wrist conventional radiographs of 388 patients from six medical centers, with pixel-level instance segmentation annotations for 618 images and SvdH BE scores for 800 images. This dataset can potentially support a wide range of research tasks related to RA, including joint space narrowing (JSN) progression quantification, BE detection, bone deformity evaluation, and osteophyte detection. It may also be applied to other wrist-related tasks, such as carpal bone fracture localization. We hope this dataset will significantly lower the barrier to research on wrist RA and accelerate progress in CAD research within the RA-related domain.
