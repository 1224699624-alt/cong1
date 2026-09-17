# Round 1 Independent Review

**Overall**: 7.40/10  
**Verdict**: REVISE

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.0 |
| Method Specificity | 7.0 |
| Contribution Quality | 7.0 |
| Frontier Leverage | 8.0 |
| Feasibility | 6.0 |
| Validation Focus | 8.0 |
| Venue Readiness | 7.0 |

主要问题：九区 latent 无可辨识监督；独立 classifier/NLL/masked reconstruction 目标堆叠；raw relation-likelihood gate 会屏蔽严重异常；violation map 反馈训练会产生循环/二阶问题；仅 synthetic corruption 与真实模型 soft prediction 存在域差。Round 1 refinement 已分别改为 candidate-triggered local evidence、统一 class-conditional posterior、reliability-only gate、audit-only stop-gradient violation map 和 original-train OOF prediction training。

