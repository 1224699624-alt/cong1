# RAM O 先验与原始 NPY 对齐及困难位置审计（2026-09-28）

实际数据源 `G:/gutou/RAM-W600`，只读 train425/val69，未使用 test 或更新模型。494 份 NPY 均为 14 通道 `int8` 二值标签，尺寸与同名 BMP 一致；真实重叠为 `sum(members)>=2`。图像/标签同步 `_R` 翻转，±4px 平移诊断最优为零偏移，没有发现标签或 O 全局坐标错误。

冻结 image-only O generator（SHA `e7b738e0b413f9138660475062e76e5d67d9df5c94afc0a2c2597ff0cf126ae5`）在 val 全部真实重叠的 O≥0.5 召回为 60.33%、precision 82.18%，但 C0-H0 真漏检重叠成员只覆盖 10,750/71,335=15.07%；单骨前景有 146,391 个 O 假高亮像素。train425 的 H0 覆盖同样只有 14.67%。

将原始 GT 按生成器训练目标缩至 192 再还原，val GT 重叠 Dice 仍 0.979711；H0 中 93.42% 的软目标≥0.5，实际 O 仅 15.07%。所以不能把失败简单归因于 `.npy` 标签错误或监督目标缩小后消失。H0 更靠近 GT 重叠边缘，O 高值常在附近，却缺少精确困难位置的强响应；O 是一张不分 14 成员身份的重叠图，也不能独自指出漏掉的骨成员。生成器原验证与下游的 sigmoid/插值顺序略有不一致，但 H0 跨阈值仅 78 个成员位置，不是主要原因。

唯一下一项验证：从同一起点比较原 BCE 与仅对 train H0 真正重叠位置加权的生成器训练，在 val 固定相同单骨误高亮预算检验 H0 召回；本轮未执行新训练。

[完整诊断](../../outputs/ram_prior_geometry_20260928/DIAGNOSIS_AND_NEXT_ACTION.md) · [GT/O 空间统计](../../outputs/ram_prior_geometry_20260928/VAL_O_GEOMETRY.json) · [训练目标重建](../../outputs/ram_prior_geometry_20260928/RESOLUTION_VAL.json) · [示例图](../../outputs/ram_prior_geometry_20260928/VAL_EXAMPLE_PANELS.png)
