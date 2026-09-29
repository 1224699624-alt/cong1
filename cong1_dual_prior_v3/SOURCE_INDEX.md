# 固定快照来源索引

仓库：`1224699624-alt/cong1`；审计 commit：`737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353`。

以下为实际读取的相关文件；“局部读取”不等于完整文件审计。权重路径/指标来自记录，不代表本轮重新加载或运行。

## S01 · main 分支
main 分支元数据：固定此次审计快照。范围：分支元数据。

[固定快照来源](https://api.github.com/repos/1224699624-alt/cong1/branches/main)

## S02 · research-wiki/experiments/r351_dual_prior_nnunet.md
用户主线、历史双数据集 pilot 与限制。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/research-wiki/experiments/r351_dual_prior_nnunet.md)

## S03 · research-workflow/refine-logs/R351_EXPERIMENT_PLAN_20260917.md
实验计划、原生参考、局部监督、5%探索门槛。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/research-workflow/refine-logs/R351_EXPERIMENT_PLAN_20260917.md)

## S04 · research-workflow/refine-logs/R351_TRACKER.md
原生权重 SHA、实际成熟初始化、数据数目与现状。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/research-workflow/refine-logs/R351_TRACKER.md)

## S05 · research-workflow/refine-logs/R351_EXPERIMENT_CODE_REVIEW.md
成熟初始化、MSD失败率、chunk身份与post-hoc guard。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/research-workflow/refine-logs/R351_EXPERIMENT_CODE_REVIEW.md)

## S06 · outputs/artifact_bundles/r351_20260917/provenance.json
历史代码/先验/成熟权重哈希和环境记录。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/outputs/artifact_bundles/r351_20260917/provenance.json)

## S07 · outputs/artifact_bundles/r351_20260917/RESUME.md
归档与下一步 native-start 四组消融。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/outputs/artifact_bundles/r351_20260917/RESUME.md)

## S08 · scripts/r351_dual_prior.py
先验/输入适配/输出refiner与 local_terms，S*(1-O)依赖。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/r351_dual_prior.py)

## S09 · scripts/nnunet_trainers/nnUNetTrainerR351DualPrior.py
TSRS冻结成熟骨干、两通道输入与固定损失调用。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/nnunet_trainers/nnUNetTrainerR351DualPrior.py)

## S10 · scripts/train_r351_overlap_prior.py
RAM真实重叠先验训练、软覆盖目标、TSRS弱迁移声明。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/train_r351_overlap_prior.py)

## S11 · scripts/r349_parameter_gradient_gate.py
参数空间梯度路由及None坐标处理；副本经Git blob核验。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/r349_parameter_gradient_gate.py)

## S12 · scripts/r350_unified_interaction_loss.py
状态支持、局部项和mass weighting定义。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/r350_unified_interaction_loss.py)

## S13 · scripts/r351_data.py
RAM先验与原图对齐、canonical方向和padding。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/r351_data.py)

## S14 · scripts/run_r201_unified_eval.py
R201指标函数及默认test入口；仅审计前290行。范围：局部读取 [1, 290]。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/run_r201_unified_eval.py)

## S15 · scripts/evaluate_r201_single_split.py
显式单split评价、skip/resize容错和expected count。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/evaluate_r201_single_split.py)

## S16 · scripts/evaluate_r329_ram_official_metrics_val.py
R329成员/overlap/pair指标、空样本和MSD失败率。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/evaluate_r329_ram_official_metrics_val.py)

## S17 · .gitignore
大权重、数据、tensor与归档的Git排除规则。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/.gitignore)

## S18 · scripts/train_r325_ram_nnunet_native_resolution_overlap_iem.py
RAM native loader和plain训练路径；读取1–220行。范围：局部读取 [1, 220]。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/train_r325_ram_nnunet_native_resolution_overlap_iem.py)

## S19 · scripts/train_r284_ramw600_overlap_prior.py
RAM PlainConvUNet适配、15原始输出/14独立成员；读取1–210行。范围：局部读取 [1, 210]。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/train_r284_ramw600_overlap_prior.py)

## S20 · AGENTS.md
项目规则、数据隔离和工作流。范围：完整文本。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/AGENTS.md)

## S21 · scripts/train_r351_ram_dual_prior.py
已见部分：student/teacher forward、native gate与固定seam ROI；未审完完整训练文件。范围：工具返回的部分文本，完整训练循环未审完。

[固定快照来源](https://github.com/1224699624-alt/cong1/blob/737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353/scripts/train_r351_ram_dual_prior.py)

