"""Render matched nnU-Net baseline versus R272 prior+loss panels on original-val."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TILE = (360, 480)

def find_image(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = root / f"{stem}{ext}"
        if p.exists() and p.stat().st_size:
            return p
    raise FileNotFoundError(stem)

def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))

def mask(path: Path) -> np.ndarray:
    return gray(path) > 0

def resize(a: np.ndarray, is_mask: bool = False) -> Image.Image:
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    return Image.fromarray(np.asarray(a, dtype=np.uint8)).convert("RGB").resize(
        TILE, Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR)

def overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    base = np.stack([image] * 3, axis=-1).astype(np.float32)
    fp, fn = pred & ~gt, gt & ~pred
    base[fp] = 0.35 * base[fp] + 0.65 * np.array([255, 30, 30])
    base[fn] = 0.35 * base[fn] + 0.65 * np.array([255, 220, 20])
    return resize(np.clip(base, 0, 255).astype(np.uint8))

def crop_box(inst: np.ndarray, first: int, second: int) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero((inst == first) | (inst == second))
    if not len(xx):
        return 0, 0, inst.shape[1], inst.shape[0]
    margin = max(24, int(round(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1) * .45)))
    return (max(0, int(xx.min())-margin), max(0, int(yy.min())-margin),
            min(inst.shape[1], int(xx.max())+margin+1), min(inst.shape[0], int(yy.max())+margin+1))

def select(path: Path, n: int) -> list[dict[str, str]]:
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["gap_bin"] in {"1-2", "3-4"}]
    by_image: dict[str, dict[str, str]] = {}
    for r in rows:
        old = by_image.get(r["image"])
        key = (float(r["baseline_pair_merged"]), float(r["baseline_gap_fp_rate"]), -float(r["gap_px"]))
        if old is None or key > (float(old["baseline_pair_merged"]), float(old["baseline_gap_fp_rate"]), -float(old["gap_px"])):
            by_image[r["image"]] = r
    return sorted(by_image.values(), key=lambda r: (-float(r["baseline_pair_merged"]), -float(r["baseline_gap_fp_rate"]), float(r["gap_px"]), r["image"]))[:n]

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis/val"))
    p.add_argument("--gt-dir", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis/val_labels"))
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--improved-dir", type=Path, default=Path("outputs/visualizations/r272_seam_prior_loss_local_original_val/masks"))
    p.add_argument("--pair-csv", type=Path, default=Path("outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations/r273_r274_matched_original_val"))
    p.add_argument("--improved-is-nnunet", action="store_true", help="read improved nnU-Net label PNG and keep class 1 only")
    p.add_argument("--num-cases", type=int, default=12)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    if "clean-test" in str(a).lower() or "articular" in str(a).lower(): raise RuntimeError("Epiphysis original-val only")
    rows, manifest = select(a.pair_csv, a.num_cases), []
    for r in rows:
        name, stem = r["image"], Path(r["image"]).stem
        image = gray(find_image(a.image_dir, stem)); inst = np.asarray(Image.open(a.gt_dir / name)); inst = inst[..., 0] if inst.ndim == 3 else inst
        gt = inst > 0
        bpath = a.baseline_dir / f"val_{name}"; ipath = a.improved_dir / name
        baseline = mask(bpath)
        if a.improved_is_nnunet:
            ipath = a.improved_dir / f"val_{name}"
            improved = gray(ipath) == 1
        else:
            improved = mask(ipath)
        if not (image.shape == gt.shape == baseline.shape == improved.shape): raise RuntimeError(f"shape mismatch: {name}")
        x0, y0, x1, y1 = crop_box(inst, int(r["instance_i"]), int(r["instance_j"]))
        cols = [(resize(image), "Original X-ray"), (resize(gt, True), "Ground truth"),
                (resize(baseline, True), "Original nnU-Net"), (resize(improved, True), "Prior + loss"),
                (overlay(image, gt, baseline), "Baseline FP/FN"), (overlay(image, gt, improved), "Improved FP/FN")]
        z = image[y0:y1, x0:x1]; zg = gt[y0:y1, x0:x1]; zb = baseline[y0:y1, x0:x1]; zi = improved[y0:y1, x0:x1]
        zcols = [(resize(z), "Original X-ray"), (resize(zg, True), "Ground truth"), (resize(zb, True), "Original nnU-Net"), (resize(zi, True), "Prior + loss"), (overlay(z, zg, zb), "Baseline FP/FN"), (overlay(z, zg, zi), "Improved FP/FN")]
        header, row_h, width = 72, 520, len(cols) * TILE[0]; canvas = Image.new("RGB", (width, header + 2*row_h), "white"); draw = ImageDraw.Draw(canvas); font = ImageFont.load_default()
        draw.text((10, 8), f"{stem} | pair {r['instance_i']}-{r['instance_j']} | gap={float(r['gap_px']):.2f}px", fill="black", font=font)
        for i, (full, label) in enumerate(cols):
            draw.text((i*TILE[0]+8, 42), label, fill="black", font=font); canvas.paste(full, (i*TILE[0], header)); canvas.paste(zcols[i][0], (i*TILE[0], header+row_h))
        draw.text((8, header+4), "Full image", fill="black", font=font); draw.text((8, header+row_h+4), "Close-gap zoom", fill="black", font=font)
        out = a.output_dir / f"{stem}_comparison.png"; canvas.save(out, optimize=True)
        manifest.append({"image": name, "panel": str(out), "pair": [int(r["instance_i"]), int(r["instance_j"])], "gap_px": float(r["gap_px"]), "crop_xyxy": [x0,y0,x1,y1]})
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rendered": len(manifest), "output_dir": str(a.output_dir)}, indent=2))

if __name__ == "__main__": main()
