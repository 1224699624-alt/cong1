# Confidence-Weighted Loss Review Summary

**Rounds**: 3  
**Final Score**: 9.27/10  
**Verdict**: READY

| Round | Score | Main issue | Resolution |
|---|---:|---|---|
| 1 | 6.93 | reliability 连乘、BCE/GCE 重复、EMA anchor、ranking 捷径、loss 堆叠 | 收缩为 piecewise base + pair-weighted hard-negative BCE |
| 2 | 8.85 | pair loss 被全背景 mean 稀释，多 pair 聚合未定义 | 有界 max 聚合并按 selected weight mass 归一化 |
| 3 | 9.27 | 无阻塞项 | READY |

最终方法严格满足：prior 只调系数，hard target 只有原 bone mask，无合法 `y=0` 时跳过，不生成 seam GT。

