# R317 / R332 / R350 与原版通用 nnU-Net 对比

这里按三组实验分别对比原版通用模型骨架。不同 split 不混排：TSRS 使用 96 例 original-val/R201；R332 使用 RAM-W600 locked test 124 例；R350 RAM 使用 validation69，R350 TSRS 使用 original-val96。

## 一、R317 TSRS seam prior vs 原版 nnU-Net

R317 采用原图+连续 seam heatmap 两通道输入 nnU-Net，并加入 continuous separation loss，alpha=0.035。原版为 R320 表中的 nnU-Net v2 2D baseline；单 seed，±为逐 case sample SD。

| 指标 | 原版 nnU-Net | R317 | R317 - 原版 |
|---|---:|---:|---:|
| Dice | 0.8971±0.0673 | 0.9029±0.0505 | +0.0058 |
| IoU | 0.8192±0.0963 | 0.8264±0.0760 | +0.0072 |
| Precision | 0.8865±0.0978 | 0.9093±0.0654 | +0.0228 |
| Recall | 0.9162±0.0505 | 0.9018±0.0619 | -0.0144 |
| Boundary IoU | 0.2399±0.0876 | 0.2452±0.0838 | +0.0053 |
| Boundary F1 | 0.3790±0.1139 | 0.3867±0.1076 | +0.0077 |
| Surface Dice 2px | 0.6545±0.1647 | 0.6684±0.1529 | +0.0139 |
| Surface Dice 5px | 0.8965±0.1181 | 0.9133±0.0887 | +0.0169 |
| HD95 px | 54.34±211.79 | 14.01±57.88 | -40.33 |
| ASSD px | 14.17±61.70 | 4.05±15.41 | -10.13 |
| Gap FP | 0.2055±0.0911 | 0.1817±0.0907 | -0.0238 |
| Merge rate | 0.5729±0.4973 | 0.5000±0.5026 | -0.0729 |
| Component MAE | 5.67±25.38 | 1.85±2.40 | -3.82 |

R317 是成熟 TSRS seam 锚点。由于历史 baseline 推理链后来被发现与 matched native recheck 不一致，这张历史表用于表示成熟结构效果，不能单独当严格 native 因果证明。

## 二、R332 RAM overlap prior vs 原版 nnU-Net

R332 使用 R325 native-resolution nnU-Net，在输出端加入14成员逐实例 refiner、own/other-member 信息、uncertainty、image-gradient、两步迭代及 surface/volume 目标。原版为 R325 plain native nnU-Net；以下是 locked test124例。

| 指标 | 原版 R325 | R332 step2 | Δ |
|---|---:|---:|---:|
| Overall DSC | 0.979362 | 0.979884 | +0.000523 |
| Overall IoU | 0.960003 | 0.961000 | +0.000998 |
| Overall NSD2px | 0.888434 | 0.894604 | +0.006170 |
| Overall MSD px | 1.307574 | 1.130006 | -0.177568 |
| Overall RAVD | 0.019481 | 0.019258 | -0.000223 |
| Overlap DSC | 0.870637 | 0.874531 | +0.003895 |
| Overlap IoU | 0.772134 | 0.778223 | +0.006089 |
| Overlap NSD2px | 0.780356 | 0.789704 | +0.009349 |
| Overlap MSD px | 1.909904 | 1.804907 | -0.104997 |
| Overlap RAVD | 0.058705 | 0.059987 | +0.001282 |
| Pair DSC | 0.828841 | 0.831425 | +0.002584 |
| Pair IoU | 0.733197 | 0.738069 | +0.004872 |
| Pair NSD2px | 0.776738 | 0.784847 | +0.008109 |
| Pair MSD px | 1.948267 | 1.691503 | -0.256764 |
| Pair RAVD | 0.351319 | 0.330240 | -0.021079 |
| Pair MSD fail rate | 0.003846 | 0.007692 | +0.003846 |

R332 是目前最清晰的成熟“重叠先验 vs 原版通用 nnU-Net”结果。RAVD 和 Pair MSD fail rate 的副作用必须保留。

## 三、R350 unified loss / gradient coordination vs 原版 nnU-Net

R350 使用统一 separation/overlap/preserve 状态损失和参数空间梯度协调；它不是 native-start fresh A/B/C/D。

### R350 TSRS

| 指标 | 原版 matched native | R350 | Δ |
|---|---:|---:|---:|
| Dice | 0.903663 | 0.903646 | -0.0000175 |
| IoU | 0.827698 | 0.827664 | -0.0000334 |
| Precision | 0.896838 | 0.908452 | +0.011614 |
| Recall | 0.915901 | 0.904145 | -0.011756 |
| Boundary IoU | 0.243221 | 0.245767 | +0.002546 |
| Boundary F1 | 0.384112 | 0.387467 | +0.003355 |
| Surface Dice2px | 0.663343 | 0.669197 | +0.005854 |
| Surface Dice5px | 0.908994 | 0.912502 | +0.003509 |
| HD95 px | 16.963342 | 16.585257 | -0.378085 |
| ASSD px | 4.381480 | 4.268398 | -0.113083 |
| Gap FP | 0.214855 | 0.184152 | -0.030703 |
| Merge rate | 0.635417 | 0.562500 | -0.072917 |
| Component MAE | 2.166667 | 1.958333 | -0.208333 |

TSRS R350 的结构指标改善，但 Dice/IoU/Recall 略降，因此不是通过严格区域 gate 的最终方法。

### R350 RAM

| 指标 | 原版 R325 native | R350 epoch29 | Δ |
|---|---:|---:|---:|
| Overall DSC | 0.978858 | 0.979409 | +0.000552 |
| Overall IoU | 0.959101 | 0.960136 | +0.001036 |
| Overall NSD2px | 0.887262 | 0.892635 | +0.005373 |
| Overall MSD px | 1.169285 | 1.079362 | -0.089924 |
| Overall RAVD | 0.020263 | 0.020029 | -0.000234 |
| Overlap DSC | 0.873731 | 0.877765 | +0.004034 |
| Overlap IoU | 0.777345 | 0.783591 | +0.006247 |
| Overlap NSD2px | 0.780398 | 0.788474 | +0.008076 |
| Overlap MSD px | 1.946378 | 1.808961 | -0.137417 |
| Pair DSC | 0.833630 | 0.836959 | +0.003329 |
| Pair IoU | 0.738825 | 0.744837 | +0.006012 |
| Pair NSD2px | 0.782834 | 0.789107 | +0.006273 |
| Pair MSD px | 1.742422 | 1.654358 | -0.088064 |
| Pair RAVD | 0.666188 | 0.652443 | -0.013745 |
| Pair MSD fail rate | 0.001992 | 0.001992 | 0 |

RAM R350 是目前最完整的“原版通用 nnU-Net → 统一损失/梯度协调”结果；相对 R332 validation anchor 的增量只有 Overall DSC +0.0000206、Overlap DSC +0.000759、Pair DSC +0.001383，应称小幅增量。

## 结论

R317 是成熟 TSRS 单骨缝结果，R332 是成熟 RAM 单骨重叠结果，R350 是统一状态损失/梯度协调在两个原版骨架上的实验，其中 RAM 结果支持小幅有效、TSRS 结果伴随区域代价。三组数字不能拼成一个双先验结果；R351 和 Round3 D 仍单独保留。
