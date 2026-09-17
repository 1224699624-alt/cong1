# Final Proposal: HA-PDSP + PCR-Loss

当前主线不再使用反向骨缝标签或 max-min bridge energy 作为训练目标。

冻结 HA-PDSP 只输出候选对系数；PCR-Loss 保留原 bone mask 作为唯一 hard target：

```text
noisy bone mask
-> annotation-side reliability
-> weighted Dice + piecewise BCE/GCE

developmental pair prior
-> stop-gradient pair coefficient
-> reweight original y=0 hard negatives between candidate bones
```

若原标签在两骨之间没有任何 `y=0`，该 pair 严格跳过，不生成伪 seam。

完整方案：`research-workflow/refine-logs/FINAL_CONFIDENCE_WEIGHTED_LOSS_PROPOSAL_20260713_144452.md`。最终独立复评 `9.27/10，READY`。本阶段未运行实验。
