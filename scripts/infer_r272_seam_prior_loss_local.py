"""Export R272 best-checkpoint original-val masks, seam maps and panels."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from run_r272_seam_prior_loss_local import SeamDataset, find_image, suppressed
from train_instance_separation_segmenter import InstanceSeparationUNet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/experiments/r272_seam_prior_loss_local_full/best.pt"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/visualizations/r272_seam_prior_loss_local_original_val"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    p.add_argument("--target-root", type=Path, default=Path("outputs/targets/r271_instance_seam_full"))
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--suppress-weight", type=float, default=0.35)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--visual-count", type=int, default=12)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); a.output_root.mkdir(parents=True, exist_ok=True)
    ds = SeamDataset(a, "val", 0, False); loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.num_workers)
    payload = torch.load(a.checkpoint, map_location=a.device)
    model = InstanceSeparationUNet(int(payload.get("args", {}).get("base_channels", 16))).to(a.device)
    model.load_state_dict(payload["model"]); model.eval()
    mask_dir, seam_dir = a.output_root / "masks", a.output_root / "seam_probability"
    mask_dir.mkdir(parents=True, exist_ok=True); seam_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for index, batch in enumerate(loader):
        name = str(batch["name"][0]); stem = Path(name).stem
        with torch.no_grad():
            out = model(batch["image"].to(a.device)); prob = suppressed(out, a.suppress_weight)[0, 0].cpu().numpy(); seam = torch.sigmoid(out["sep"])[0, 0].cpu().numpy()
        label = np.asarray(Image.open(a.raw_root / a.dataset / "val_labels" / name), dtype=np.uint8)
        gt = label > 0; h, w = gt.shape
        pred = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR) >= a.threshold
        seam_full = cv2.resize(seam, (w, h), interpolation=cv2.INTER_LINEAR)
        Image.fromarray(np.uint8(pred * 255)).save(mask_dir / name)
        Image.fromarray(np.uint8(np.clip(seam_full, 0, 1) * 255)).save(seam_dir / name)
        if index < a.visual_count:
            image_path = find_image(a.raw_root, a.dataset, "val", stem)
            gray = np.asarray(Image.open(image_path).convert("L")).astype(np.float32) / 255.0
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
            rgb = np.stack([gray, gray, gray], -1)
            gt_boundary = gt ^ cv2.erode(gt.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            pred_boundary = pred ^ cv2.erode(pred.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            p1 = (rgb * 255).astype(np.uint8); p1[gt_boundary] = (0, 255, 0)
            p2 = (rgb * 255).astype(np.uint8); p2[gt_boundary] = (0, 255, 0); p2[pred_boundary] = (255, 80, 40)
            p3 = (rgb * 255).astype(np.uint8); p3[seam_full > 0.5] = (255, 0, 255)
            gt_rgb = np.asarray(Image.open(a.raw_root / a.dataset / "val_labels" / name).convert("RGB").resize((w, h)))
            canvas = np.concatenate([np.asarray(Image.open(image_path).convert("RGB").resize((w, h))), gt_rgb, p1, p2, p3], axis=1)
            Image.fromarray(canvas).save(a.output_root / f"{stem}_comparison.png", optimize=True)
    (a.output_root / "manifest.json").write_text('{"run_id":"R272_infer_original_val","clean_test_used":false,"checkpoint":"' + str(a.checkpoint).replace('\\', '/') + '"}', encoding="utf-8")
    print(f"wrote {len(ds)} masks to {mask_dir}")


if __name__ == "__main__": main()
