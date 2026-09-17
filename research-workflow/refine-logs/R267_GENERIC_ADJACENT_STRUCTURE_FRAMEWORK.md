# R267：面向相邻发育结构的通用先验与分离损失

## 设计目标

R266 的 `CC/CM/CR` 是儿童腕手解剖专用关系，不适合迁移到牙齿。R267 将先验对象
从“关系类别”改成“局部相邻结构对”：模型不需要知道两个结构是腕骨、掌骨还是牙齿，
只需要知道它们在当前图像中可能相邻、之间存在可分离的间隙。

```text
候选结构实例/中心
        ↓
局部邻接图（距离、方向、尺度、重叠风险）
        ↓
每条边生成 gap map + 两侧 support map + confidence
        ↓
通用分离损失
```

## 先验通道协议

输入保持 `1 + 2K` 通道：

```text
channel 0       X-ray
channel 1,2     pair 0: gap, support
channel 3,4     pair 1: gap, support
...
channel 2K-1,2K pair K-1: gap, support
```

每个 pair 的元数据只包含：候选结构索引、局部中心距离、尺度比、方向差、图像谷值
证据和置信度。禁止写入 CC、CM、CR、牙齿编号等解剖类别，以便同一损失迁移。

年龄/性别/牙龄可以作为可选条件向量，影响邻接边的先验置信度和 abstention，不改变
损失的数学形式。低龄骨化中心或未萌出牙齿可以通过低置信度/空槽自然退出，而不是
被错误地当成缺失结构或错误关系。

## 通用分离损失

对每个有效邻接边 `e`：

\[
L_e=c_e\{\operatorname{softplus}[(p_{gap,e}-\tau_e)/T_g]
 + \lambda_s\operatorname{softplus}[(s_e-p_{support,e})/T_s]\}.
\]

其中：

- `p_gap,e` 是 gap map 中的前景概率；
- `p_support,e` 是两侧结构支持区中的前景概率；
- `c_e` 是几何/图像/条件先验置信度；
- `tau_e = tau_0 + delta(1-c_e)`，不确定边界使用更宽上限；
- 所有结构共享同一套参数，不再有 CC/CR 等关系分支。

后续可增加通用边界保留项：在 support 的内侧边界带上约束预测前景不低于局部结构
置信度，防止分离损失通过削薄骨/牙边界来降低 gap 概率。

## 掌骨与牙齿的适配方式

两种数据只需更换 `AdjacentStructureAdapter`：

| 适配器 | 候选结构 | 邻接边来源 | 条件变量 |
|---|---|---|---|
| HandAdapter | 掌骨/腕骨候选中心或实例 | kNN + Delaunay + 谷值 | 骨龄、性别 |
| DentalAdapter | 牙胚/牙冠候选实例 | 同侧相邻牙序 + kNN | 年龄、性别、牙龄 |

训练器只读取统一的 gap/support/confidence 通道，因而不需要为掌骨和牙齿各写一套
损失函数。不同图像大小通过局部 pair 坐标和尺度归一化处理，不要求整幅图空间对齐。

## 当前代码状态

已新增 [nnUNetTrainerR267AdjacentStructure.py](/G:/gutou/YOLO+SAM/scripts/nnunet_trainers/nnUNetTrainerR267AdjacentStructure.py)。
它实现了关系无关的 pair loss，并记录 `valid_slots`、`gap_excess`、
`support_probability` 等通用诊断量。当前仍使用 Dataset206 的通道布局做接口 sanity，
尚未把旧 R265 的 CC/CR 先验宣称为通用先验，也尚未启动 R267 训练。

下一步应先写 HandAdapter 的 generic pair manifest，再用同一 trainer 做掌骨 pilot；
牙齿数据接入时只需生成同样的 `1+2K` 通道和 manifest，不能直接复用掌骨坐标规则。
