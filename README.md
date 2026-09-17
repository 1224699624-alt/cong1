# YOLO + SAM 掌骨区域分割

本项目采用两阶段方案完成掌骨区域分割：

1. 用 YOLO 检测骨骺或关节面区域。
2. 将 YOLO 检测框作为通用 SAM 的提示，生成最终分割 mask。

当前本地版本已经支持将以下改进加入 SAM 提示与微调流程中：

- 背景负提示点
- 边界附近采样
- 环状/结构化背景约束
- 将 prompt 学习纳入微调，而不只是训练 `mask_decoder`
- 提示点热图监督
- 边界前景/背景对比约束

这份 README 只描述本地代码当前支持的实验，不涉及服务器端是否已同步。

## 项目结构

```text
YOLO+SAM/
  checkpoints/                         # SAM 和 YOLO 权重
  configs/                             # YOLO 数据集配置
  data/
    raw/                               # 原始 TSRS_RSNA 数据
    yolo/                              # 转换后的 YOLO 检测数据
  outputs/                             # 推理结果、评估结果、SAM 微调输出
  runs/                                # Ultralytics 训练输出
  scripts/                             # 数据准备、训练、推理、评估、汇总脚本
  run_*.sh                             # Linux 批量实验脚本
```

## 研究工作流

本项目已预留一套面向 `Codex` / ARIS 风格研究的项目内工作流入口，方便后续围绕这个仓库持续做文献调研、idea 设计、实验计划、自动复盘与写作交接。

关键位置：

```text
AGENTS.md                      # 项目级研究代理说明
tools/install_aris_yolosam.ps1 # 安装 ARIS-Codex 技能到本项目
tools/update_aris_yolosam.ps1  # 更新/对齐已安装技能
research-workflow/             # 研究工作区与模板
```

安装项目级 ARIS-Codex 技能：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\install_aris_yolosam.ps1
```

安装完成后，建议优先使用这些工作流入口：

```text
/research-wiki init
/research-lit "bone age epiphysis segmentation"
/idea-discovery "improve YOLO+SAM epiphysis segmentation beyond current refiner baselines"
/experiment-bridge "research-workflow/refine-logs/EXPERIMENT_PLAN.md"
/research-pipeline "new segmentation direction for this repository"
```

## 环境安装

如果你在 Linux 服务器上使用 `bash`：

```bash
source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu
```

如果你在 `/bin/sh` 下：

```sh
. /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu
```

安装依赖：

```bash
pip install -r requirements.txt
```

验证 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 1. 生成 YOLO 检测数据

骨骺数据：

```bash
python scripts/prepare_yolo_dataset.py --dataset TSRS_RSNA-Epiphysis --padding-ratio 0.08 --overwrite
```

关节面数据：

```bash
python scripts/prepare_yolo_dataset.py --dataset TSRS_RSNA-Articular-Surface --padding-ratio 0.08 --overwrite
```

## 2. 训练 YOLO

训练骨骺 YOLO：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_yolo.py --data configs/TSRS_RSNA-Epiphysis.yaml --model yolov8l.pt --imgsz 1024 --epochs 100 --batch 4 --device 0 --name epiphysis_yolov8l_1024
```

训练关节面 YOLO：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_yolo.py --data configs/TSRS_RSNA-Articular-Surface.yaml --model yolov8l.pt --imgsz 1024 --epochs 100 --batch 4 --device 0 --name articular_surface_yolov8l_1024
```

YOLO 输出默认位于：

```text
runs/detect/outputs/yolo/
```

如果 Ultralytics 自动加后缀，例如：

```text
runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt
```

后续推理时请使用真实存在的 `best.pt` 路径。

## 3. 下载通用 SAM 权重

```bash
python scripts/download_sam_checkpoint.py
```

默认输出：

```text
checkpoints/sam_vit_b_01ec64.pth
```

## 4. 零微调推理

### 4.1 仅使用 box prompt

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_yolo_sam.py --dataset TSRS_RSNA-Epiphysis --split test --yolo-weights runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --imgsz 1024 --conf 0.20 --box-padding-ratio 0.05 --prompt-mode box --save-overlays
```

### 4.1.2 错误类型感知动态修正

这组实验仍然使用通用 SAM，不微调模型。它会对同一张图尝试多个 `box-padding-ratio`，再根据候选 mask 的稳定性、mask/box 面积一致性和越框比例自动选择结果，用来同时缓解缺漏和过度分割：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_yolo_sam.py --dataset TSRS_RSNA-Epiphysis --split test --yolo-weights runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --imgsz 1024 --conf 0.20 --prompt-mode box --dynamic-correction --dynamic-padding-ratios 0.00,0.03,0.05,0.08,0.10,0.15 --dynamic-base-padding-ratio 0.05 --save-overlays --out-root outputs/ablations/zero_shot_dynamic_box_consistency_conf020
```

评估：

```bash
python scripts/evaluate_masks.py --dataset TSRS_RSNA-Epiphysis --split test --pred-dir outputs/ablations/zero_shot_dynamic_box_consistency_conf020/TSRS_RSNA-Epiphysis/test/masks
```

### 4.1.3 Mask-to-Prompt 二次修正

这组实验借鉴 SAMRefiner 的思路：先用 YOLO box 得到 SAM 初始 mask，再从初始 mask 自动生成二次 box、前景点、边界附近背景点，并把第一次 SAM 的低分辨率 mask logit 作为 `mask_input` 再送回 SAM。模型仍然使用通用 SAM，不做微调。

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_yolo_sam.py --dataset TSRS_RSNA-Epiphysis --split test --yolo-weights runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --imgsz 1024 --conf 0.20 --box-padding-ratio 0.05 --prompt-mode box --mask-to-prompt-refine --refine-box-padding-ratio 0.03 --refine-num-positive-points 1 --refine-num-negative-points 4 --refine-negative-dilate-kernel 9 --save-overlays --out-root outputs/ablations/zero_shot_mask_to_prompt_refine_conf020_pad005
```

评估：

```bash
python scripts/evaluate_masks.py --dataset TSRS_RSNA-Epiphysis --split test --pred-dir outputs/ablations/zero_shot_mask_to_prompt_refine_conf020_pad005/TSRS_RSNA-Epiphysis/test/masks
```

如果要一次性跑完 mask-to-prompt refinement 的参数消融，直接执行：

```bash
./run_epiphysis_mask_to_prompt_ablation.sh
```

脚本会自动比较以下因素：

- 二次 box 的 `refine-box-padding-ratio`
- 边界附近负点数量 `refine-num-negative-points`
- 负点环带大小 `refine-negative-dilate-kernel`
- 是否使用第一次 SAM 的低分辨率 `mask_input`

汇总结果默认输出到：

```text
outputs/ablations/TSRS_RSNA-Epiphysis_test_mask_to_prompt_refine_summary.md
```

跑完上述候选后，可以继续做候选 mask 融合消融。该步骤不重新运行 SAM，只融合已经生成的候选 mask：

```bash
./run_epiphysis_mask_fusion_ablation.sh
```

融合结果默认输出到：

```text
outputs/ablations/TSRS_RSNA-Epiphysis_test_mask_fusion_summary.md
```

## 错误解耦残差修正网络

如果要继续冲击更高 Dice，可以训练一个轻量的 under/over-segmentation aware residual refiner。它会把 `GT` 与 mask-to-prompt refinement 结果对比，自动构造：

- `missing_map = GT & ~refined_mask`
- `excess_map = refined_mask & ~GT`

模型输入包括 X-ray 原图、zero-shot SAM mask、mask-to-prompt refined mask、候选 mask 的 union/intersection、边界图和距离图；输出最终 mask，同时辅助预测 missing/excess 两类错误区域。

一键流程：

```bash
./run_epiphysis_error_refiner.sh
```

这个脚本会先为 `train/val/test` 生成候选 mask，再训练 residual refiner，最后在 test 上推理和评估。它比前面的零训练实验更耗时，但这是当前最有希望继续提升到 `0.89+` 的方向。

补充：

- `scripts/infer_yolo_sam.py` 现在支持 `--refine-iters`，可以把 mask-to-prompt refinement 从两轮扩展成多轮迭代，更接近 SAMRefiner 的使用方式。
- `scripts/train_error_refiner.py` 现在支持 `--min-epochs` 和 `--patience`，用于在验证集长期不提升时提前停止训练。
- `scripts/select_error_refiner_threshold.py` 会在 `val` 上自动扫描 threshold，并把最佳阈值记录到 `outputs/error_refiner/<dataset>/threshold_sweep_val.json`。

如果要直接比较两条 refinement 路线，可以执行：

```bash
./run_epiphysis_refiner_ablation.sh
```

这个脚本会比较：

- `zero_shot_box_only_conf020_pad005`
- `zero_shot_mask_to_prompt_refine_iter3_pad003_neg2_k5`
- `error_refiner_iter3_unet512_b32`

并输出汇总到：

```text
outputs/ablations/TSRS_RSNA-Epiphysis_test_refiner_path_summary.md
```

### 4.2 使用 box + 环状背景负点

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_yolo_sam.py --dataset TSRS_RSNA-Epiphysis --split test --yolo-weights runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --imgsz 1024 --conf 0.20 --box-padding-ratio 0.05 --prompt-mode box+neg --num-negative-points 8 --negative-point-offset-ratio 0.08 --save-overlays --out-root outputs/predictions_boxneg
```

### 4.3 使用 box + 前景/背景点提示

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_yolo_sam.py --dataset TSRS_RSNA-Epiphysis --split test --yolo-weights runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --imgsz 1024 --conf 0.20 --box-padding-ratio 0.05 --prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08 --save-overlays --out-root outputs/predictions_boxfgbg
```

## 5. 评估结果

评估命令：

```bash
python scripts/evaluate_masks.py --dataset TSRS_RSNA-Epiphysis --split test --pred-dir outputs/predictions/TSRS_RSNA-Epiphysis/test/masks
```

如果你更改了 `--out-root`，请将 `--pred-dir` 指向对应实验目录下的 `masks` 子目录。

## 6. Prompt 消融参数说明

以下新参数主要在两个脚本中使用：

- `scripts/infer_yolo_sam.py`
- `scripts/train_sam.py`

### `--prompt-mode`

可选值：

- `box`
  - 只使用 box prompt
- `box+neg`
  - 使用 box prompt + 结构化背景负点
- `box+fgbg`
  - 使用 box prompt + 前景正点 + 结构化背景负点

### `--num-positive-points`

只在 `box+fgbg` 模式下有意义。  
控制每个实例使用多少个前景正点。

当前默认：

```text
1
```

### `--num-negative-points`

控制每个实例使用多少个背景负点。  
这些负点不是随机采样，而是按轮廓顺序均匀分布，形成环状约束。

当前默认：

```text
8
```

### `--negative-point-offset-ratio`

控制背景负点相对轮廓外扩的距离比例。  
值越小，负点越贴近边界；值越大，负点越远离目标。

当前默认：

```text
0.08
```

### `--prompt-loss-weight`

点级监督损失权重。  
在启用结构化点提示时，会对提示点位置施加额外监督。

当前默认：

```text
0.25
```

### `--prompt-heatmap-sigma`

控制提示点热图的高斯半径。  
热图越大，局部监督越平滑。

当前默认：

```text
4.0
```

### `--prompt-heatmap-loss-weight`

提示点热图监督权重。  
用于在 mask 学习之外，额外约束模型在提示点附近的局部行为。

当前默认：

```text
0.15
```

### `--contrastive-loss-weight`

边界前景/背景对比损失权重。  
用于拉开前景核心区域与边界外背景环带的特征差异，减少过分割。

当前默认：

```text
0.05
```

### `--contrastive-temperature`

边界对比损失的温度参数。

当前默认：

```text
0.1
```

### `--boundary-kernel-size`

构造边界环带时使用的核大小。  
用于从前景 mask 生成外侧背景环带和内侧前景核心区域。

当前默认：

```text
5
```

## 7. 微调 SAM

如果你已经有骨骺 YOLO 权重，不想重新训练 YOLO，只想直接微调骨骺 SAM，那么建议从这里开始。

当前默认复用的骨骺 YOLO 权重：

```text
runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt
```

### 7.1 仅微调 `mask_decoder`

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sam.py --dataset TSRS_RSNA-Epiphysis --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --epochs 50 --lr 1e-5 --num-workers 16 --max-instances 64 --cache-embeddings --prompt-mode box --output outputs/sam_finetune/TSRS_RSNA-Epiphysis/best_mask_decoder.pt
```

### 7.2 使用背景负点 + 热图监督 + 边界对比约束

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sam.py --dataset TSRS_RSNA-Epiphysis --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --epochs 20 --lr 5e-6 --num-workers 16 --max-instances 64 --box-padding-ratio 0.05 --box-jitter-ratio 0.08 --cache-embeddings --prompt-mode box+neg --num-negative-points 8 --negative-point-offset-ratio 0.08 --prompt-loss-weight 0.25 --prompt-heatmap-loss-weight 0.15 --contrastive-loss-weight 0.05 --output outputs/sam_finetune/TSRS_RSNA-Epiphysis/ablations/maskdec_boxneg_heat_contrast_e20_lr5e-6_pad005_jit008.pt
```

### 7.3 使用前景/背景点 + `prompt_encoder` + 热图监督 + 边界对比约束

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sam.py --dataset TSRS_RSNA-Epiphysis --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --epochs 20 --lr 5e-6 --num-workers 16 --max-instances 64 --box-padding-ratio 0.05 --box-jitter-ratio 0.08 --cache-embeddings --prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08 --train-prompt-encoder --prompt-loss-weight 0.25 --prompt-heatmap-loss-weight 0.15 --contrastive-loss-weight 0.05 --output outputs/sam_finetune/TSRS_RSNA-Epiphysis/ablations/maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008.pt
```

### 7.4 使用前景/背景点 + `prompt_encoder` + `GBC adapter` + 热图监督 + 边界对比约束

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sam.py --dataset TSRS_RSNA-Epiphysis --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --epochs 20 --lr 5e-6 --num-workers 16 --max-instances 64 --box-padding-ratio 0.05 --box-jitter-ratio 0.08 --cache-embeddings --prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08 --train-prompt-encoder --prompt-loss-weight 0.25 --prompt-heatmap-loss-weight 0.15 --contrastive-loss-weight 0.05 --use-gbc --output outputs/sam_finetune/TSRS_RSNA-Epiphysis/ablations/maskdec_prompt_gbc_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008.pt
```

## 8. 加载微调后的权重推理

示例：使用“前景/背景点 + `prompt_encoder` + 热图监督 + 边界对比约束”训练得到的 checkpoint 推理

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_yolo_sam.py --dataset TSRS_RSNA-Epiphysis --split test --yolo-weights runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt --sam-checkpoint checkpoints/sam_vit_b_01ec64.pth --finetuned-checkpoint outputs/sam_finetune/TSRS_RSNA-Epiphysis/ablations/maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008.pt --imgsz 1024 --conf 0.20 --box-padding-ratio 0.05 --prompt-mode box+fgbg --num-positive-points 1 --num-negative-points 8 --negative-point-offset-ratio 0.08 --save-overlays --out-root outputs/ablations/maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008
```

评估：

```bash
python scripts/evaluate_masks.py --dataset TSRS_RSNA-Epiphysis --split test --pred-dir outputs/ablations/maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008/TSRS_RSNA-Epiphysis/test/masks
```

## 9. 骨骺消融实验

本地已经为骨骺准备了一组 prompt 消融实验，脚本是：

```text
run_epiphysis_ablation.sh
```

当前脚本会顺序跑以下实验：

### 零微调组

1. `zero_shot_box_only_conf020_pad005`
   - 只用 box prompt 的零微调基线
2. `medsam_zero_shot_box_only_conf020_pad005`
   - 将通用 SAM 换成 MedSAM，不微调，直接复用 YOLO 框推理评估
3. `zero_shot_dynamic_box_consistency_conf020`
   - 通用 SAM + 动态 padding 候选选择，根据 box-mask 一致性自动修正疑似缺漏或外扩
4. `zero_shot_mask_to_prompt_refine_conf020_pad005`
   - 通用 SAM + mask-to-prompt 二次修正，从初始 mask 反推二次 box、正点、负点和 mask prompt
5. `zero_shot_box_neg_ring_conf020_pad005`
   - box + 环状背景负点
6. `zero_shot_box_fgbg_ring_conf020_pad005`
   - box + 前景正点 + 环状背景负点

### 微调组

7. `maskdec_boxonly_e20_lr5e-6_pad005_jit008`
   - 只训练 `mask_decoder`
8. `maskdec_boxneg_heat_contrast_e20_lr5e-6_pad005_jit008`
   - `mask_decoder` + 背景负点 + 提示点热图监督 + 边界对比约束
9. `maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008`
   - `mask_decoder + prompt_encoder` + 前景/背景点 + 热图监督 + 边界对比约束
10. `maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008_gtbox`
   - 与第 9 组使用同一份 checkpoint，但推理时改用 `GT box`
   - 用于判断误差主要来自 YOLO 框，还是来自 prompt 学习本身
11. `maskdec_prompt_gbc_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008`
   - 在第 9 组基础上加入 `GBC adapter`

## 10. 自动汇总消融结果

跑完后可自动汇总为 Markdown 表格：

```bash
python scripts/summarize_ablation_metrics.py --dataset TSRS_RSNA-Epiphysis --split test
```

## MedSAM 零微调对照

如果要把通用 SAM 换成 MedSAM，并且不微调 MedSAM，只复用之前训练好的 YOLO 框直接推理评估，按下面三步跑：

```bash
python scripts/download_medsam_checkpoint.py
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_yolo_medsam.py --dataset TSRS_RSNA-Epiphysis --split test --yolo-weights runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt --medsam-checkpoint checkpoints/medsam_vit_b.pth --imgsz 1024 --conf 0.20 --box-padding-ratio 0.05 --save-overlays --out-root outputs/ablations/medsam_zero_shot_box_only_conf020_pad005
```

```bash
python scripts/evaluate_masks.py --dataset TSRS_RSNA-Epiphysis --split test --pred-dir outputs/ablations/medsam_zero_shot_box_only_conf020_pad005/TSRS_RSNA-Epiphysis/test/masks
```

`run_epiphysis_ablation.sh` 已经把这组实验加入默认流程，实验名是：

```text
medsam_zero_shot_box_only_conf020_pad005
```

汇总默认输出：

```text
outputs/ablations/TSRS_RSNA-Epiphysis_test_ablation_summary.md
```

## 11. 设计动机

这次 prompt 扩展的核心目标是把以下思想加入项目并做消融：

- 背景负提示点
- 边界附近采样
- 环状/结构化背景约束
- 训练时把 prompt 学习纳入微调，而不是只训练 `mask_decoder`
- 用点热图监督替代单纯点级约束
- 用边界前景/背景对比约束减少过分割

对应实现方式如下：

1. 背景负提示点
   - 推理时在目标外侧构造结构化负点
2. 边界附近采样
   - 训练时从真实 mask 轮廓附近采样，而不是只按 box 几何关系采样
3. 环状/结构化背景约束
   - 训练和推理都尽量保持点沿轮廓顺序分布
4. prompt 学习纳入微调
   - 训练时将点提示送入 `prompt_encoder`
5. 提示点热图监督
   - 在提示点邻域施加局部高斯热图监督
6. 边界前景/背景对比约束
   - 使用前景核心区域和边界外背景环带做特征对比

## 注意事项

- YOLO 负责定位，SAM 负责分割。
- 如果 YOLO 漏检，SAM 无法恢复完全缺失的区域。
- `--conf` 建议尝试 `0.10`、`0.20`、`0.30`。
- `--box-padding-ratio` 建议尝试 `0.00`、`0.05`、`0.08`、`0.10`、`0.15`。
- 如果只训练 `mask_decoder`，优先推荐开启 `--cache-embeddings`。
- 如果使用 `--train-image-encoder-last-n`，不要再加 `--cache-embeddings`。
- 做对比实验时，建议每组使用独立的 `--out-root`，避免覆盖旧结果。
