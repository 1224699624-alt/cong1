# R351：nnU-Net统一双先验pilot

用户目标：两个数据集都使用输入骨缝先验、可替换通用模型、输出重叠先验、统一协调机制。本轮只用nnU-Net；不要求Dice/IoU大幅提升，但不应退化，主结构指标应有明显改善。

计划：[R351_EXPERIMENT_PLAN_20260917](../../research-workflow/refine-logs/R351_EXPERIMENT_PLAN_20260917.md)
审查：[R351_EXPERIMENT_CODE_REVIEW](../../research-workflow/refine-logs/R351_EXPERIMENT_CODE_REVIEW.md)
记录：[R351_TRACKER](../../research-workflow/refine-logs/R351_TRACKER.md)
资产：`outputs/artifact_bundles/r351_20260917/`。

输入侧新增受限零初始化适配器；输出侧复用RAM成员补全网络。RAM多标签训练图像条件overlap prior，再冻结迁移TSRS。TSRS原始索引PNG不能表达真实多重归属，因此真实overlap标签状态为unknown，不能将前景补全代理当作重叠GT。

TSRS2轮pilot已完成，96例R201：Dice0.903888861、IoU0.828067391、gapFP0.192697431、merge0.572916667。相比匹配原生区域不退化，gapFP/merge分别相对下降约10.31%/9.84%；相比初始化R350，gapFP和merge回退。完整架构总收益与本轮增量收益必须区分。

RAM真实审计通过，2轮pilot已完成，最佳epoch1：Overall DSC0.979199379、IoU0.959733948、Overlap MSD1.829829581、Pair MSD1.654147634，相对native gate通过；相对R332 anchor有部分整体/NSD退化。TSRS positive-seam-guard仅为post-hoc validation诊断，结果另存，未用于test选择。未用test调参，最终结果以单checkpoint绑定JSON为准。
