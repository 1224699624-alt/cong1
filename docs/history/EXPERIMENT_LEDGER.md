# 历史实验账本 (Experiment Ledger)

> **保护与追溯说明**：本账本记录 2026-09-29 仓库清洗前所有重要实验族、研究分支及关键诊断结果。
> 原始完整代码、输出产物与历史记录可通过保护点标签 `pre-cleanup-2026-09-29` 或 commit `118fe1f` 完整回溯。

---

## 一、核心实验与关键决策汇总

| 实验/实验族 ID | 状态 | 数据集 | 核心结论与关键指标 | 停止/不再继续原因 | 后继版本/替代方案 | 原始证据/关联文档 |
|---|---|---|---|---|---|---|
| **09-29 C00 vs CS** | `NOT_PROMOTED` | RAM | holdout Overlap Dice: C00 `0.90556392`, CS `0.90556490`；val C00 `0.87730192`, CS `0.87730057` (val 差异 `−0.00000136`，区间包含 0)。固定 H0 恢复均为 0。 | CS 相比 C00 无实际稳定增益，且对正确成员有极轻微损失；不能启动大规模 SO 联合或长训。 | 转向真实 C0 假连接处的 S 质量与代价审计；冻结主线。 | `research-wiki/experiments/ram_so_heatmap_20260929.md` |
| **09-29 O hard-weight** | `TRADEOFF / INCONCLUSIVE` | RAM | 候选相对原 O 多覆盖 66 个 holdout H0 像素、26 个 val；但全图 AP 与普通重叠有代价，holdout 总指标未超原 O。 | 空间重加权干预弱，存在指标权衡，未展现明确晋级优势；不作为新标准 O。 | 保持 C00/CS 架构；暂缓 O 单独长训。 | `outputs/ram_so_heatmap_20260929/SO_PRIOR_RESULT.md` |
| **09-29 S/O 统一接口** | `PASS (验证)` | RAM | 737 参数 post-R332-step2 头 (21→32→1)，末层 zero-init，signed bound 8；合成与真实病例零增量和反向传播检验全部通过。 | 原型工程通过验证，但后续分割增量需取决于先验本身信息质量。 | 现役条件修正模块协议基础。 | `codex_handoff/RAM_SO_HEATMAP_CONDITIONAL_HANDOFF/02_DUAL_HEATMAP_UNIFIED_CORRECTOR.md` |
| **high-res P0/P1 读出** | `INCONCLUSIVE / NOT_PROMOTED` | RAM | val69 自然 Q 的 P0/P1 AP 为 0.123940 vs 0.119111；精确 H0 选中 26,981 vs 26,975 (全量 71,335)；精确 FP 16.5w+。 | P1 相比 P0 无稳定增量，且背景负例代价失控；不具备启动完整高分辨率分割修正的依据。 | 转向数据/原图盲态复核，不启动新网络模块。 | `research-wiki/experiments/ram_highres_readout_20260928.md` |
| **O 接入位置对照** | `FAIL / NOT_PROMOTED` | RAM | test124 step2 Overlap Dice: `ZERO 0.874588`, `INPUT_O 0.873789`, `REFINER_O 0.873814`；Overlap NSD 亦均低于 ZERO。 | 无论在 nnU-Net 输入端还是 Refiner 输入端，引入 O 均劣于等算力零先验；证明单纯改变接入位置无效。 | 彻底停止在骨架/Refiner 入口重复搜索接入位置；转为局部条件增量。 | `research-wiki/experiments/ram_o_position_test_20260928.md` |
| **O 先验几何与分辨率** | `DIAGNOSTIC` | RAM | 评估了不同分辨率下 O 先验在重叠骨块边界的几何覆盖特性。 | 诊断任务完成，明确了小骨骼边缘受采样影响的定量界限。 | 纳入 S/O 统一热力图设计依据。 | `research-wiki/experiments/ram_o_prior_geometry_20260928.md` |
| **H0 final-logit 监督** | `FAIL / NOT_PROMOTED` | RAM | Overlap Dice 从 source 的 0.877535 降至 0.877496；val69 固定 H0 真恢复从 36 降至 13；两组留出恢复均为 0。 | 产生梯度但无目标收益，未能解决漏检瓶颈，反而增加成员 FN。 | 放弃 final-logit 局部正监督长训；转向 S/O 独立条件修正。 | `research-wiki/experiments/ram_h0_final_target_20260924.md` |
| **Source 梯度与敏感性诊断** | `FAIL / NOT_PROMOTED` | RAM | 目标 source BCE 改善 overlap DSC 0.000199，但降低 NSD 0.000054，未恢复初始 71,337 个最难 val 成员。 | 未通过预设性能阈值 (Gate Fail)。 | 停止沿此路径调优。 | `research-workflow/RESEARCH_BRIEF.md` |
| **GT-Negative Guard** | `FAIL / NOT_PROMOTED` | RAM | 负例保护减少 FP 2441 像素，但同时增加 FN 2404 像素；难例恢复依然为 0，overlap NSD 下降 0.000056。 | 正负样本冲突，虽然总像素 FP 减少但边界结构恶化。 | 不晋级、不扩展负例保护损失。 | `outputs/ram_negative_guard_20260922/DIAGNOSIS_AND_ACTION.md` |
| **Identity-Context 因果探测** | `FAIL / NOT_PROMOTED` | RAM | 850 步对齐对照，规范序列 vs 打乱序列 Overlap DSC 增量仅为 0.000027 (0.0027pp)；置信区间包含 0。 | 身份上下文未提供实质分割增益，71335 难例恢复依然为 0。 | 停止该分支调参与长训。 | `outputs/ram_identity_causal_20260922/DIAGNOSIS_AND_ACTION.md` |
| **White-Box AdamW 优化状态** | `INCONCLUSIVE` | RAM | 恢复历史 AdamW 优化器动量状态未能通过性能闸门。 | 历史动量恢复未显示出超越 fresh AdamW 的优势。 | 统一使用带 refiner tail LR 的 fresh AdamW 作为后续训练基准。 | `outputs/whitebox_adam_20260922/DIAGNOSIS_AND_ACTION.md` |
| **R351 双先验 nnU-Net** | `HISTORICAL_EXPLORATION` | RAM / TSRS | TSRS 96 例 R201: Dice 0.90389, IoU 0.82807, gapFP 0.1927；RAM: DSC 0.9792, IoU 0.9597。相比 R350 gapFP/merge 回退。 | 非 clean native-start 成功证据，多重先验耦合导致训练不稳定。 | 演进为 V3 / V3.1 协议及后续解耦的 S/O 分步实验。 | `research-wiki/experiments/r351_dual_prior_nnunet.md` |
| **R350 协调参考线** | `HISTORICAL_REFERENCE` | RAM | 组合协调证据，探索骨缝与重叠先验的协同机制。 | 属于组合证据而非独立因果，参数量偏大。 | 演化为 V3.1 协议。 | `research-workflow/refine-logs/R350_TRACKER.md` |
| **R332 骨骼成员与重叠基线** | `HISTORICAL_ANCHOR` | RAM | 成熟的 2-step refiner，在 RAM holdout 上取得稳定 Overlap Dice (~0.9056)。 | 作为当前所有 RAM 实验的核心冻结基准 (C0 / C00)。 | 当前 RAM 主线的基础支撑模型。 | `research-wiki/experiments/ram_so_heatmap_20260929.md` |
| **R317 骨缝先验成熟锚点** | `HISTORICAL_ANCHOR` | TSRS | TSRS clean_test_v2 上 Dice 0.910029 (对比 ARAA 标杆 0.911766)。 | 属于成熟基线，但 TSRS 标签语义有噪声，非当前匹配重点。 | 保留为 TSRS 侧历史对比锚点。 | `research-workflow/RESEARCH_BRIEF.md` |
| **R268~R270 腕骨与边界分类** | `SUPERSEDED` | TSRS / RAM | 尝试使用 ROI 裁剪和独立边界二分类器解决腕骨粘连。 | 局部 ROI 切分破坏全局上下文，收益未能推广。 | 转向全局热力图先验与多标签分割。 | `outputs/analysis/` 历史归档 |
| **R188~R198 MaskDINO 系列** | `FAIL / SUPERSEDED` | TSRS | 尝试将通用检测分割大模型引入腕骨检测与分割任务。 | 预训练模型在微小骨缝与复杂粘连处严重过拟合，单图及小批量过拟合测试无法泛化到真实测试集。 | 放弃通用检测/查询架构，回归医学专用网络 nnU-Net 与高分辨率 Refiner。 | `outputs/analysis/r190~r198_*` |
| **R110~R134 难例清洗与协议** | `HISTORICAL` | TSRS | 建立了 clean_test_v2 评估切片及初始难例集合。 | 任务目标已达成，切片标准已固化。 | 固化为标准评测规范 `clean_test_v2`。 | `outputs/analysis/r122~r134_*` |
| **R001~R038 早期 YOLO+SAM** | `SUPERSEDED` | TSRS | 基于 YOLO 检测框作为 SAM / MedSAM / HQ-SAM 的 Prompt 进行边界细化。 | 框 Prompt 无法有效传达细长骨缝与接触边缘的语义，过平滑且易导致相邻骨骼合并。 | 演进为像素级语义分割与双先验精细修正架构。 | 早期根目录代码与 `outputs/error_refiner/` |

---

## 二、从历史标签恢复代码或数据

所有被清理的旧文件并未从 Git 历史中消除。如果需要完全恢复某项实验当时的源码、日志或产物：

```bash
# 1. 查看清洗前保护点的状态
git checkout pre-cleanup-2026-09-29

# 2. 或者从特定历史 commit 中单独检出某一个文件或文件夹：
git checkout pre-cleanup-2026-09-29 -- outputs/bridge_logs/...
git checkout pre-cleanup-2026-09-29 -- scripts/train_r351_dual_prior.py
```
