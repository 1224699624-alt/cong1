# V3 第一轮审计报告

本轮遵循接手提示与 `cong1_dual_prior_v3/` 中的 TASK/REPO_AUDIT/ACCEPTANCE_TESTS。没有启动新的正式训练；此前已完成的 v2 T0 短跑仅保留其日志，不作为本轮新结论。

## 已定位的本地资产

详细 checkpoint keys、大小和 SHA 写入 [ASSET_LOCK.json](ASSET_LOCK.json)。当前两个 native 文件实际存在：

- TSRS：`outputs/nnunet/r317_image_centers_only/r202_two_channel_zero_prior.pth`，SHA `55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b`。
- RAM：`outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth`，SHA `88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9`。
- TSRS seam manifest：`outputs/priors/r317_image_centers_only/manifest.json`。
- RAM seam manifest：`outputs/priors/r323_ram_native_r317_single_seed/manifest.json`。
- `/root/r351_workspace/overlap/full/best.pth` 是服务器历史运行时资产，不在本地工作区；不能把它写成已锁定的本地核心权重。

旧 R202/R317/R320、R323/R325、R327–R333、R349–R351 的代码与 refine logs 均可在本地找到；历史 tracker、pilot 和 checkpoint 保留。

## 当前实际前向接线

TSRS 当前 final wrapper 为：

```text
image + seam
  → StructuralPriorPlugin / SeamInputAdapter
  → native nnU-Net 输入 [adapted_image, zero placeholder]
  → native logits
  → OutputCompletionPrior（若启用）
  → final logits
```

旧 R351 TSRS 仍是 `ImageOverlapPrior(image,seam)`、`SeamInputAdapter(image,seam,overlap)` 和 `OutputCompletionPrior` 的组合。它的 output interaction 使用 `normalize(seam) * (1-overlap)`，所以 seam 清零会切断一部分 overlap context；这不能作为 clean overlap-only 证据。新 final plugin 已有独立 `overlap_probability` 接口，但 image-only overlap generator 尚未实现。

RAM 当前 final runner 为：

```text
RAM image + RAM seam map
  → optional input SeamInputAdapter
  → RAM native 14-member nnU-Net logits
  → ImageOverlapPrior / OutputCompletionPrior
  → final 14-member logits
```

RAM 的成员输出保持 14 个独立 sigmoid 通道；不能改成互斥 softmax。R325 `plain_native` checkpoint 用 `model` state，历史 R351/R332 refiner 只可作为带来源的成熟 output 初始化，不能冒充 fresh native model。

## 标签和监督来源

TSRS train/val 971 张标签已完成 P-mode audit，first-channel collision 为 0；118 张出现重复索引的断开连通域，可安全按 connected components 拆分并保持 binary union。TSRS 现有导出标签是 union/匿名索引，尚未发现可证明完整 multi-membership overlap 的原始标注来源；因此真实 overlap state 仍为 `unknown`。不能将前景 union、颜色相邻、crowding 或漏分写成真实 overlap GT。

RAM mask 是 14 个独立成员通道；真实 overlap 定义为已观测成员数至少 2。R327–R332 和 RAM overlap prior 训练资产提供真实 multi-membership supervision，可继续核验其 split/coverage。

TSRS seam 优先使用 R317；RAM seam 优先使用 R323。四个先验的来源、覆盖率、可观测性仍需分别写入 `PRIOR_SEMANTICS_TSRS.json` 和 `PRIOR_SEMANTICS_RAM.json`，目前尚未宣称这一步完成。

## R349 梯度反例

本地 `scripts/r349_parameter_gradient_gate.py` 与 V3 evidence 的 source blob 一致。对 `base=(1,1)`、`aux=(-1,None)`，旧实现会得到 projected auxiliary 与 base 点积约 `-0.5`，仍有冲突；V3 需要把缺失坐标按完整参数空间的 0 处理。该缺陷已复现，尚未修改历史 R349 文件。下一步应新增隔离版修复和 regression test，再接入 core runner。

## 硬停/未决项

- 当前没有依据启动新的 A/B/C/D 正式训练；接手提示要求第一轮先完成审计。
- clean C 组仍需要 image-only overlap generator 或明确标记 legacy coupled context，不能只把 seam 置零。
- TSRS overlap 真实监督缺口仍存在；可做 weak transferred evidence，但不能冒充 native overlap annotation。
- native checkpoint 已定位并在之前服务器验证过；本轮本地资产锁定记录已生成，但不重新声称本地重新跑过 baseline。
- 旧 R351 cross-dataset dual prior 保留为 optional extension，不进入 core final model。

## 建议修改文件

1. 新增 `scripts/r349_parameter_gradient_gate_v3.py`：修复 None 坐标的全空间投影，保留旧 R349 不变。
2. 新增 `scripts/tsrs_prior_semantics.py` / `scripts/ram_prior_semantics.py` 或 JSON manifest：分别记录四个先验来源与可观测性。
3. 扩展 `scripts/final_prior_plugin.py`：加入 image-only overlap generator 的明确版本；旧 coupled path 标记 legacy。
4. 在 core runner 中增加 C 信息隔离、case ID/shape 显式失败和 N0 logits/mask 对比；旧 pilot 输出不覆盖。
5. 增加 V3 验收中的 L01–L10、G01–G06、E01–E07 的最小 CPU/真实 batch 测试。

本报告没有把“尚未定位/尚未验证”写成“不存在”，也没有修改原始数据、历史结果或旧 R351 实现。
