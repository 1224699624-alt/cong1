---
type: experiment
node_id: exp:ram_overlap_fn_geometry_20260927
added: 2026-09-27
---

# RAM 重叠边界与第二成员漏检

完整证据：[DIAGNOSIS_AND_NEXT_ACTION.md](../../outputs/ram_overlap_fn_geometry_20260927/DIAGNOSIS_AND_NEXT_ACTION.md)。本轮只读 RAM val69 原始14成员标签和既有固定 H0 坐标；没有训练、test 或 TSRS。

C0 的真实 overlap 成员条件 FN 率约7.65%，非重叠正成员约0.689%。全量真实 overlap 正成员中约9.44%距自身标签边界约1px，而已有审计抽到的32,732个H0位置中约44.82%处于该边界带，说明困难漏检有明显边界集中性；该H0样本不能外推全部71,335个。严格匹配子集10,265个位置中，10,249个已经预测出一块骨，被漏第二成员的最终概率中位数约0.00761。

成熟refiner固定source累计最多加5；H0 source低于−5。此前source/final正监督、负类保护、稳定身份和显式O source-head对照均未选择性修复该集合。当前没有证据确定唯一主因或宣布标注错误。唯一优先下一步是冻结C0高分辨率特征的成员条件读出，验证困难正成员与匹配负成员是否可分，然后才考虑成员条件边界补全的小型匹配训练。
