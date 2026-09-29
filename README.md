# 掌骨与骨骺精细解剖分割 (Bone Age & Epiphysis Segmentation)

> **当前基准状态（2026-09-29）**：本项目聚焦于手骨 X 光片中复杂骨骺与腕骨接触/重叠区域的高精度解剖分割，核心方法基于 **nnU-Net 密集预测 + 骨缝先验 (S) / 重叠先验 (O) 的解剖条件修正机制**。
>
> 早期 YOLO+SAM 提示微调方案已被证明存在跨缝膨胀与过平滑缺陷，相关资料已归档至历史记录中。

---

## 1. 核心研究问题与当前结论

### 核心任务
针对儿童骨龄评估中相邻骨骼间隙微小（1~3 像素）、接触面粘连（False Merge）以及重叠区域漏检（Overlap FN）的问题，构建具备解剖形态感知能力的条件修正分割网络。

### 当前有效基线锚点 (Active Anchors)
- **RAM-W600 数据集主线**：
  - **`R332` (C0 / C00 基线)**：成熟的 2-step Refiner 架构，在 RAM holdout84 取得稳定 Overlap Dice **`0.9056`**。当前所有先验改进均以此作为冻结控制组。
- **TSRS 数据集主线**：
  - **`R317`**：骨缝先验成熟锚点，在 `clean_test_v2` 切片上达到 Dice **`0.910029`**（对照外部 ARAA 标杆 `0.911766`）。

### 最近已证伪 / 停止长训的方向 (负结论与禁忌)
为避免后续研究与自主 Agent 重复试错，以下方向**严禁盲目重启或调参**：
1. **禁止直接进行 SO 联合长训**：
   - 2026-09-29 最新 `C00 vs CS` 对照证明，引入当前 S 先验后相比 C00 的 Overlap Dice 变化量仅为 `−0.00000136`（且导致 6 个正确边界微小损失）。当前 S 先验尚未展现宏观增益。
2. **禁止继续在骨架输入/Refiner 入口搜索 O 先验接入位置**：
   - 2026-09-28 test124 实测证明，`INPUT_O (0.873789)` 和 `REFINER_O (0.873814)` 均劣于等算力的零先验 `ZERO (0.874588)`。先验信息质量本身是瓶颈，换入口无效。
3. **禁止对 H0 局部正监督长训**：
   - 2026-09-24 实验表明，直接在 14 成员 final-logit 上监督 H0 像素导致 Overlap Dice 下降 0.003925pp，漏检不降反增。
4. **不启动全网络高分辨率特征读出分割**：
   - 2026-09-28 高分辨率读出检验显示，P1 相比 P0 无稳定增量，且背景负例代价失控（FP 高达 16.5 万）。
5. **早期 YOLO+SAM 系列已彻底终止**：
   - 边界框 Prompt 无法表达细长骨缝语义，已全面转向专用密集分割与多标签轻量修正头。

---

## 2. 当前唯一步伐与执行入口 (Next Action)

### 当前唯一推荐下一步
1. **定向审计 RAM S 先验在真假连接处的纯度**：
   - 在真实 C0 预测出现假连接（False Connection）的具体坐标处，盲态抽检 S 先验的响应强度，确认其是否真正具备阻断粘连的能力。
2. **热力图语义净化**：
   - 保持 C0/R332 冻结，先从数据和先验生成器内部改善 S 与 O 的信噪比，再考虑模型修正。

### 推荐阅读顺序
1. **最新现状与总结**：[`research-wiki/experiments/ram_so_heatmap_20260929.md`](research-wiki/experiments/ram_so_heatmap_20260929.md)
2. **交接规格与边界**：[`codex_handoff/RAM_SO_HEATMAP_CONDITIONAL_HANDOFF/00_README_AND_START.md`](codex_handoff/RAM_SO_HEATMAP_CONDITIONAL_HANDOFF/00_README_AND_START.md)
3. **历史实验全景与被否决路线**：[`docs/history/EXPERIMENT_LEDGER.md`](docs/history/EXPERIMENT_LEDGER.md) 与 [`docs/history/SUPERSEDED_ROUTES.md`](docs/history/SUPERSEDED_ROUTES.md)

---

## 3. 仓库结构导航

```text
/
├─ README.md                         # 本文档（当前主线、有效锚点与导航）
├─ AGENTS.md                         # 本地 Coding Agent 执行工作流与安全规则
├─ requirements.txt                  # Python 依赖清单
├─ configs/                          # 实验与数据集配置
├─ scripts/                          # 核心可复用代码库（模型定义、评估器、数据管道）
├─ launchers/                        # 当前有效且允许执行的实验启动脚本
│  ├─ ram/                           # RAM 数据集当前有效启动脚本（09-28/09-29 校验入口）
│  └─ tsrs/                          # TSRS 数据集有效复现脚本（R317 等）
├─ research-wiki/                    # 结构化实验记录与各阶段核心结论摘要
│  └─ experiments/                   # 各项关键实验的详细记录文档
├─ research-workflow/                # 研究工作流与阶段性 Brief（RESEARCH_BRIEF.md）
├─ codex_handoff/                    # 面向 Agent 的任务交接包与运行锁模板
├─ docs/
│  └─ history/                       # 历史清洗索引与实验墓碑（去重、删除清单与恢复说明）
│     ├─ EXPERIMENT_LEDGER.md        # 历史实验总账本
│     ├─ SUPERSEDED_ROUTES.md        # 被取代/证伪的研究路线深度复盘
│     ├─ DELETION_MANIFEST.json      # 机器可读的删除产物与去重索引
│     ├─ SCRIPT_USAGE_AUDIT.csv      # 历史脚本审计与保留决策
│     └─ EXACT_DUPLICATES.csv        # 相同文件去重记录
└─ outputs/                          # 当前最新实验的必要结论报告与结果表格
```

---

## 4. 历史文件可恢复性说明

本次仓库清洗**未重写 Git 提交历史，未做 force-push**。所有早期中间产物、批量图表面板、被替换的旧脚本及运行日志已完好保存在标签 **`pre-cleanup-2026-09-29`** 中。
如需检出任意历史文件：
```bash
git checkout pre-cleanup-2026-09-29 -- <file-path>
```
详细恢复操作请查阅 [`docs/history/README.md`](docs/history/README.md)。
