# V3 验收与回归测试清单

这是给 Codex 实现/运行的测试规格，不是本轮已全部通过的报告。现有只实际运行了 `evidence/gradient_none_probe.json` 中的 CPU 反例，以及交付包自身检查。

## 1. 资产与基线

| ID | 验收条件 |
|---|---|
| A01 | native checkpoint 的实际路径、SHA256、state keys 和 source split 可追溯；缺失时禁止默认从头训练 |
| A02 | TSRS zero-prior 第二通道与原 R202 的等价性有数值对比；不得只凭文件名 |
| A03 | RAM `use_prior=False`，14 成员的 sigmoid 语义保持；隐藏 prior head 不进入分割/结构 loss |
| A04 | 先验全部关闭时，真实 batch 与完整验证 N0 的坐标、logits/probability、threshold mask、指标复现 |
| A05 | train/val ID 无交叉，原始标签解释一致；TSRS 1833.png/874–875 差异有明确结论，不静默漏文件 |
| A06 | N0、mature reference、native-start A/B/C/D 的身份分开，不把 R350/R332 写成原生 |

## 2. 前向信息依赖

| ID | 验收条件 |
|---|---|
| F01 | InputSeamAdapter 零初始化恒等；输出残差有界；核心输入不偷偷读取 O |
| F02 | OutputOverlapAdapter 零初始化恒等；O 有独立特征通路，不依赖 S 非零才能生效 |
| F03 | C 不读取有效 seam；包括 generator、输入占位通道、输出 context、loss 和缓存；legacy 混合路径不能进入 clean 主表 |
| F04 | C 在 S=0 时，O 的改变能够到达输出模块。零 residual 初始化阶段只检查特征通路；在测试用非零末层/一次受控更新后才检查输出敏感性，不能将初始恒等误判为失败 |
| F05 | B 不读取 O、overlap teacher 权重或弱 overlap loss；A 无任何先验依赖 |
| F06 | RAM 成员顺序与TSRS margin/softmax等价；成员批处理 chunk、TF32/AMP 策略和 checkpoint reload 可复现 |
| F07 | 推理 forward 参数/数据访问无 GT、GT实例、训练 target、per-case validation 选择；允许 frozen 图像先验 |
| F08 | 冻结 backbone 时输入 adapter 梯度可穿过 backbone；teacher no_grad 不影响学生路径 |

## 3. 局部监督

| ID | 验收条件 |
|---|---|
| L01 | 人工构造同一张图同时含可信 background seam 和真实多成员 overlap，两项 loss 均生效，不按 dataset/image 二选一 |
| L02 | TSRS 只有 union GT 时 K_overlap=unknown；foreground 不能直接当 overlap，单输出通道也不是完整成员集合 |
| L03 | 部分成员缺失：两个已知阳性可确认 overlap；不足两个阳性且其他未知不能监督为阴性 |
| L04 | I_seam/I_overlap 支持合法且在 valid 内；soft weight 非零支持严格属于对应 bool mask |
| L05 | padding/ignore/未观测成员不贡献 loss 或面积归一化；改变 padding 大小不改变相同有效区域的局部均值/质量权重 |
| L06 | 未观测真实 overlap loss 无梯度贡献，但前向 overlap 分支仍存在；弱 union 项单独命名和记录，不能冒充真实成员监督 |
| L07 | weak 支持来自冻结 teacher、detach、固定规则，不由当前预测/错误或不同 arm 动态选择 |
| L08 | 任一空状态 loss 为有限、可求导的零；不会除零、NaN，或用空状态伪造“已无结构错误” |
| L09 | 关闭 arm 的结构项时，preserve 分区随之正确更新，不残留隐藏监督 |
| L10 | raw S 和 O 同时高时保留 conflict 记录；只对定义的监督支持 assert 互斥，不强迫原先验互斥 |

## 4. 梯度协调

G01 必需反例：base=(1,1)、aux=(-1,None)；投影后与 base 点积 >= -数值容差。旧 R349 会失败，详见证据文件。

G02：base 和 aux 均完整连接、完全一致、完全反向、部分断连、全零、无辅助项、weight=0、base norm≈0；全部有限。负权重被拒绝。

G03：同一 trainable 参数顺序贯穿 autograd、投影、assign_gradients；有冻结参数时不发生错位。

G04：individual cap 和 aggregate cap 定义明确，各项以及合成 auxiliary/base ratio 满足其契约；不能把三项各0.15说成合计0.15。

G05：seam/overlap 梯度方向、与 base 的方向及更新前后状态均记录；不把梯度级测试当作验证集不退化证明。

G06：D_no_routing 与 D 除 routing 开关外完全一致（包括初始化、局部 masks/权重、seed、optimizer、预算、teacher 和 selection）。

## 5. 空间与 evaluator

E01：RAM `_R` 图像/GT canonical flip 一次，先验不二次翻；image/S/O/GT/valid 同步变换。

E02：完整 case ID 一致；缺预测、额外预测、尺寸/方向错位显式失败；不能靠旧 wrapper 的 skip/resize 放行。

E03：R201 frozen 函数在已知同一 mask 上返回原值；不调用默认 clean-test main。不把前景背景带称精确骨缝GT。

E04：构造漏掉一个组件的反例，证明 component_merge_rate 不能单独区分漏分与粘连；组合 Recall/计数/可视化报告。

E05：RAM MSD 均值变好但 fail rate 升高时质量 gate 失败；GT空 pair 的排除不静默改动，报告 denominator。

E06：baseline metric=0、undefined/null/NaN、缺 case、不同 checkpoint 拼表、同一病例被当多个独立病人等反例被拒绝或明确标 unavailable。

E07：区域 margin、探索结构门槛、ROI、train pair list、threshold 和 selection 在新结果之前冻结；TSRS unknown overlap 指标为 N/A，不填0。

## 6. 真数据 smoke 与结果级验收

两数据集各至少一个真实 batch 测到：forward/backward/update/reload、末层有效学习信号、有限显存、无 GT 推理；日志标注真实 case ID 和 source hashes。不要要求零初始化首步所有隐藏层都有非零梯度。

完整 pilot 之后，分别输出区域 gate、主结构 gate、次要结构代价和 relative-to-N0/A/B/C 的差异。未满足科研收益条件应标 `SCIENTIFIC_NEGATIVE_OR_INCONCLUSIVE`，而不是伪造 PASS、改指标或改论文主线。
