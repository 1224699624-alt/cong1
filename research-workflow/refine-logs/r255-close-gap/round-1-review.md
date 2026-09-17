# Round 1 Review

**Overall score**: 6.88/10  
**Verdict**: MAJOR REVISION / NOT READY  
**Drift warning**: 主线保持，但存在漂移为全手 component-count、推理期骨龄切割器和通用前景收缩的风险。

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 9.0 |
| Method Specificity | 6.0 |
| Contribution Quality | 7.0 |
| Frontier Leverage | 7.5 |
| Feasibility | 6.0 |
| Validation Focus | 5.5 |
| Venue Readiness | 5.5 |
| Overall | 6.88 |

## Blocking findings

1. 跨病例实例 ID/骨对身份没有闭环；建议删除语义 i/j，改用无身份局部描述符。
2. 1–2 px y=0 不能自动视为可靠；必须 full-resolution、增强后存活、图像证据强、robust/capped loss。
3. same_pred_component 是阈值化动态门控；建议用 detached continuous bridge risk。
4. 骨龄训练/推理协议矛盾；首版应明确为 training-only privileged information。
5. X-ray valley/double-edge 需要局部归一化公式。
6. rank loss 存在过切风险；首版删除，先验证 anchor 定位。
7. 低骨龄真正风险是错误碎裂、小中心消失，而不是新增中心。
8. gap width 必须同时报告原图、预处理及相对骨宽归一化尺度。

## Recommended simplification

- 图内 connected-component pair；
- 局部位置/面积比/方向/距离描述符；
- 连续骨龄只调节接近风险与 abstention；
- full-resolution 稳定 y=0 bottleneck；
- capped background BCE；
- 暂时删除 L_gap_rank；
- 不增加 Transformer、diffusion 或 dense seam decoder。

