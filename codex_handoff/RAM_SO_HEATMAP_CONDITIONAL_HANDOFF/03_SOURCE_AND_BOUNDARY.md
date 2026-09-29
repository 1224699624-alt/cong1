# 03｜来源、事实与本次新增设计的边界

日期：2026-09-29。

## 用户资料（优先事实基础）

### S1：`sources/O_LABEL_AUDIT_20260928.md`

原文件：`DIAGNOSIS_AND_NEXT_ACTION(1).md`，标题《RAM 骨重叠先验 O 与原始 NPY 标签核对》。

- “直接回答／O在哪里失效”：标签格式、图像对齐、总体与H0响应、单骨误高亮。
- “是坐标偏移…”：±4像素平移、低分辨率GT重建、EXT_分层、上采样与sigmoid顺序。
- “下一项最小验证”：提出了固定C0、对训练H0重加权的O生成器匹配实验，明确当时NOT_RUN。

本交接沿用该建议并补充train内部校准、唯一像素去重、精确权重/分母、同容量与资源约束；不把建议当作已成功结果。

### S2：`sources/RAM_CROSS_ROUND_20260927.md`

原文件：`2026-09-27--ram-overlap--r00--web-analysis-handoff.md`。

- §2：RAM split、C0哈希、optimizer/scaler、评价对象。
- §3–4：F0/F1/S1、OHEAD的结果，source范围、H0与oracle、成员边界难例。
- §7：当前不宜继续推广的分支，下一步冻结高分辨率读出仍NOT_RUN。

特别保留其边界：oracle +2.9749pp不是模型收益；C0和追加384步不是同预算因果比较；未证明唯一成因。

### S3：`sources/MATURE_MAP_20260920.md`

原文件：`2026-09-20--dual-prior--r3--round-review.md`。

- R317：TSRS成熟“输入图＋局部loss”组合，旧baseline不完全matched。
- R323/V3 B：RAM骨缝路线存在，但Round3增量极小且混合。
- R332：逐成员共享修正，图像/own/other-member上下文、两步。
- 不能把不同split与完整初始化的历史成绩拼成新CSO。

### S4：已有高分辨率读出任务

`CODEX_RAM_HIGHRES_FEATURE_READOUT_EXPERIMENT.md`是此前生成的任务规格，不是实跑报告。
其关键内容：强P0基线、单个H位置、fit/calibration/holdout、真实候选Q、低误报诊断、读出不是分割。
本包不重复附上大任务书，Codex从本地实际执行状态定向承接。新任务不得将其NOT_RUN自动改PASS。

## 必须保持分母差异

- S1的71,335是H0“成员位置”数，15.07%按该集合统计；同一唯一像素可对应多个成员。
- 总体重叠像素、单骨误高亮像素是唯一像素计数。
- S2给出H0唯一像素70,751；与成员计数分别保存。
- 先前154,910个“漏检重叠像素”的15.46%是另一个集合，不与H0的15.07%混称。
- S1的0.5降至0.2是既有诊断，不授权继续用val扫描阈值。

## 本包新增、尚未证明的部分

- 统一双热力图数据接口及元数据。
- S的RAM定向质量核验和由证据触发的改进分支。
- O的soft-grid困难权重具体式、适配留出、calibration相同误报预算、锁定工作点。
- 成熟R332后共享条件增量的默认原型、C00/CS/CO/CSO矩阵。
- 后续预算门槛、白盒记录、参数/图测试与TSRS适配范围。

这些属于提议，不保证有效，不代表本地代码已经这样实现。beta、修正范围、正式LR和步数没有经过本次训练验证。

## 少量外部方法背景（不是项目结果依据）

### R1：PyTorch官方BCEWithLogitsLoss说明

核查日期2026-09-29。官方文档说明BCE-with-logits、手动重加权及正负权重对precision/recall取舍的影响。
本包不因此要求用户升级PyTorch；本地运行版本须单独记录。

```text
https://docs.pytorch.org/docs/2.13/generated/torch.nn.BCEWithLogitsLoss.html
```

本包建议的H0空间权重是自有实验候选，不是官方推荐用于该医学任务的最优loss。

### R2：PointRend，CVPR 2020

Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick.
PointRend: Image Segmentation As Rendering.

```text
https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html
```

它为利用细粒度特征进行掩码修正提供已发表设计背景；不证明本项目高分辨率特征有效，也不要求采用其迭代采样或替换nnU-Net。

## 本次实际完成与未完成

已完成：阅读用户报告、对照已交付读出任务、编写这两份执行规格与运行锁草案、打包文档。
未完成：没有访问用户G盘/服务器，没有加载C0或先验真实权重，没有实施新O训练、P0/P1训练或CSO分割训练。
文档语法/哈希/链接检查不等于模型实验PASS。
