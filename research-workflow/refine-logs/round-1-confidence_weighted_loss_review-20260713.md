# Round 1 Independent Review

**Overall**: 6.93/10  
**Verdict**: REVISE

主要问题：多项 reliability 连乘会压制困难真标签；BCE/GCE 重复；EMA eligibility 排除真正桥接；valley ranking 可通过提高 core confidence 逃避；loss 组件过多。修订后收缩为 annotation reliability + bounded uncertainty attenuation，以及直接作用于已有背景 hard negatives 的 pair-weighted BCE；最终仅保留 `L_base + alpha L_pair`。

