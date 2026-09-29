# RAM S/O 有界改进与统一条件接口（2026-09-29）

执行新S/O交接包00–03。范围RAM train425内部fit256/cal85/holdout84及val69，未读test、未跑TSRS，未重训P0/P1或完整分割。

S为RAM R323 seed3231 epoch58生成器 + 冻结R285 proposal，raw relation×heat score；GT-free推理但非单一image-only。12固定train病例gap几何proxy AP0.466、P/R@0.5为0.885/0.168；8真实C0病例gap背景FP平均S0.0338、正确FG0.00143。独立分割价值仍CONDITIONAL_VALUE_PENDING。

O同SHA e7b738...起点的原BCE与beta1去重H0空间重加权，各2048成功步，仅更新O；C0冻结。calibration原O单骨FPR1.09046%精确匹配，锁定阈值用于holdout/val。候选相对reference多覆盖66个holdout H0唯一像素、26个val；AP与普通重叠有小代价、holdout未超过原O，状态TRADEOFF_OR_INCONCLUSIVE，不晋级新O、不宣称分割收益。重加权额外全图质量mean0.123%，干预弱的限制保留。

post-R332-step2共享21→32→1新头737参数、signed bound8，仅新末层zero-init。无F_high/身份表/Q限域/source回路；C00/CS/CO/CSO接口同形。6合成测试和8真实病例零增量/首backward通过，旧权重buffers未变，新分割训练NOT_RUN。

唯一下一优先分割草案为探索性C00 vs CS，冻结C0和R332，等预算仅训练同一新头；不是本轮已授权队列。O pair及评价服务器壁钟109秒，累计含profile/失败/检查保守约164秒，低于2GPU小时。

[本轮报告](../../outputs/ram_so_heatmap_20260929/SO_PRIOR_RESULT.md) · [修正器就绪说明](../../outputs/ram_so_heatmap_20260929/CONDITIONAL_CORRECTOR_READY.md) · [原始表](../../outputs/ram_so_heatmap_20260929/SO_PRIOR_RESULTS.csv) · [实际锁](../../outputs/ram_so_heatmap_20260929/SO_PRIOR_RUN_LOCK.json)

## 后续实际 C00 vs CS（同日）

已按上轮唯一优先pair执行，不是四组矩阵：冻结C0/R332，仅训练同一个737参数post-step2头；fit256上C00/CS各512成功步，同初始SHA、同病例及增强张量序列。holdout84/val69按本机原R329 padded画布口径配对评价，未读test。

主结果：holdout Overlap Dice C00 `0.90556392`、CS `0.90556490`；val C00 `0.87730192`、CS `0.87730057`。val CS−C00 `−0.00000136`，33患者成对描述区间`[−0.00000908,+0.00000709]`。固定H0在两组均0恢复。仅C00相对冻结C0的很小变化不能归于S。

同一CS头开关S的固定8 holdout病例只令27个成员位置跨阈值：21误报删除、6正确成员失去，说明S进入但决策作用局部。严格复现2026-09-22历史C0 JSON失败；当前服务器原legacy.collect和新padded evaluator完全一致，故主比较使用同机同协议，旧环境漂移单列。下一步不启动SO/长训，优先审计RAM S在真正C0假连接位置的覆盖和正确前景代价。

[后续完整诊断](../../outputs/ram_so_c00_cs_20260929/DIAGNOSIS_AND_RESULT.md) · [标准表](../../outputs/ram_so_c00_cs_20260929/STANDARD_RESULTS.csv) · [逐例差](../../outputs/ram_so_c00_cs_20260929/PER_CASE_DIFF.csv)
