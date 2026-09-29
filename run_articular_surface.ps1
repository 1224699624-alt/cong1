Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python scripts\prepare_yolo_dataset.py --dataset TSRS_RSNA-Articular-Surface --padding-ratio 0.08 --overwrite
python scripts\train_yolo.py `
  --data configs\TSRS_RSNA-Articular-Surface.yaml `
  --model yolov8l.pt `
  --imgsz 1024 `
  --epochs 100 `
  --batch 4 `
  --name articular_surface_yolov8l_1024

python scripts\download_sam_checkpoint.py
python scripts\infer_yolo_sam.py `
  --dataset TSRS_RSNA-Articular-Surface `
  --split test `
  --yolo-weights outputs\yolo\articular_surface_yolov8l_1024\weights\best.pt `
  --sam-checkpoint checkpoints\sam_vit_b_01ec64.pth `
  --imgsz 1024 `
  --conf 0.20 `
  --box-padding-ratio 0.05 `
  --save-overlays

python scripts\evaluate_masks.py `
  --dataset TSRS_RSNA-Articular-Surface `
  --split test `
  --pred-dir outputs\predictions\TSRS_RSNA-Articular-Surface\test\masks
