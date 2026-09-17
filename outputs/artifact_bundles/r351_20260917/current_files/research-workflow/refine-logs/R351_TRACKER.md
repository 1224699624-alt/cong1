# R351 实验记录

- 目标：nnU-Net上双先验完整接入；当前不更换骨干，不重训原生baseline。
- 服务器：connect.westc.seetacloud.com:52807；项目/root/autodl-tmp/YOLO_SAM_generic_src。
- 新输出：/root/r351_workspace（系统盘约29GB空闲），与历史数据盘实验隔离。
- TSRS原生权重：r317_image_centers_only/r202_two_channel_zero_prior.pth；SHA256 55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b。
- RAM原生权重：r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth；SHA256 88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9。
- TSRS训练初始化：R350 mature；RAM初始化：R332 steps_2；两者不是原生对照。

## 已验证

- 本地/远程共享模块4项测试通过：identity但可学习输入、unknown overlap零监督、真实multi-membership及padding、单/多通道输出identity。
- overlap模型2train+2val真实batch训练/评估通过；这只是管线检查，不是效果结论。
- TSRS新wrapper真实2例batch通过，初始化概率差异为0；标签声明unknown，未将前景当重叠。
- 远程TSRS训练划分875/96；本地原始train_labels仅874（缺1833.png），实验使用远程875集，无本地训练混入。

## 最新状态（2026-09-17）

- 独立审查完成；RAM chunk=3保持TF32身份一致；MSD失败率保护补齐。
- overlap完整6轮完成：val BCE0.062097，真实重叠区平均概率0.5363，非重叠区0.0191。选第6轮；不代表TSRS域迁移已被真实重叠标签验证。
- TSRS2轮完成，完整96例R201：Dice0.903888861、IoU0.828067391、gapFP0.192697431、merge0.572916667；原生gate通过，但相对R350骨缝指标回退。
- RAM2例identity/logit差0，反向梯度审计通过；两轮pilot已完成。最佳epoch=1，整体DSC=0.979199379、IoU=0.959733948，Overlap MSD=1.829829581，Pair MSD=1.654147634；严格native gate通过，但相对R332 anchor的Overall DSC/IoU及部分NSD退化，Pair RAVD升高。
- TSRS positive-seam-guard固定推理诊断已完成：Dice=0.903860755、IoU=0.828016984、Gap FP=0.191553841、merge=0.572916667；这是观察同一validation退化后提出的post-hoc val诊断，不是独立泛化结果，也未用于测试集选择。
- 历史RAM R350 CSV已修正21行绑定第29轮，原表已备份；旧TSRS R317失配原生比较标记不可用于matched结论。
- 初始证据、overlap权重、TSRS适配器与R201已同步到outputs/artifact_bundles/r351_20260917。

## 待完成

- 大体RAM数据和先验归档传输完成后，核对文件数/大小与哈希；保留服务器源目录。
- 未来若要形成最终因果结论，仍需native-start双先验复跑和最小消融；本轮不再启动。

任何尚未完成的训练均不得表述为已获得指标提升。
