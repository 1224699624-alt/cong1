---
type: experiment
node_id: exp:ram_intensity_hypothesis_20260927
added: 2026-09-27
---

# RAM 灰度重叠假设只读审计

完整分析：[POSITION_AND_PRIOR_DECISION.md](../../outputs/ram_intensity_hypothesis_20260927/POSITION_AND_PRIOR_DECISION.md)。仅 RAM train425 / val69；没有训练、test 或 TSRS。

真实 overlap 在完整 val69 的灰度均值为 206.61，单骨为 185.04，背景为 74.50，且 69/69 病例 overlap 平均更亮。但 train 固定的亮度阈值234在 val 的 overlap 召回仅9.54%，单骨 FP达399,509。固定困难 H0 的10,265个严格匹配对中，原始亮度 AUC约0.501，局部亮度相对控制反而降低；已有 O 的 AUC为0.841。

局部对比度加入 C0 摘要+O 的 train-fit/val-check 读出后，AUC从0.940120降至0.939074，log loss从0.356970降至0.349511。它仅显示少量概率信息，未证明真实成员恢复或分割指标改善。当前不支持简单白度二值化，也不依据先验叙事更换网络接入点。若未来获得在 H0 上更有效、且不过量误报单骨的图像线索，再测试一个高分辨率 decoder/skip 融合点的同起点对照。
