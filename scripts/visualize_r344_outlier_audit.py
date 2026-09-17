"""Create deterministic R317-vs-R344 outlier comparison panels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
IDS = ["1518", "1666", "3777", "1694", "2246"]
OUT = ROOT / "outputs" / "visualizations" / "r344_outlier_audit"


def load_gray(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    lo, hi = np.percentile(image, [1, 99])
    return np.clip((image.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(mask) > 0


def overlay(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    color_array = np.asarray(color, dtype=np.float32)
    rgb[mask] = .52 * rgb[mask] + .48 * color_array
    boundary = mask ^ ndimage.binary_erosion(mask)
    rgb[boundary] = color_array
    return np.clip(rgb, 0, 255).astype(np.uint8)


def difference_panel(gray: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    common = old & new
    removed = old & ~new
    added = new & ~old
    rgb[common] = .72 * rgb[common] + .28 * np.asarray((40, 160, 255))
    rgb[removed] = np.asarray((255, 35, 35))
    rgb[added] = np.asarray((255, 215, 0))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def crop_box(diff: np.ndarray, margin: int = 80) -> tuple[int, int, int, int]:
    labels, count = ndimage.label(diff, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0, 0, diff.shape[1], diff.shape[0]
    sizes = np.bincount(labels.reshape(-1)); sizes[0] = 0
    chosen = np.argsort(sizes)[-min(3, count):]
    ys, xs = np.where(np.isin(labels, chosen))
    x0, x1 = max(0, int(xs.min()) - margin), min(diff.shape[1], int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(diff.shape[0], int(ys.max()) + margin + 1)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, x1 = max(0, cx - side // 2), min(diff.shape[1], cx + (side + 1) // 2)
    y0, y1 = max(0, cy - side // 2), min(diff.shape[0], cy + (side + 1) // 2)
    return x0, y0, x1, y1


def tile(images: list[np.ndarray], labels: list[str], width: int = 360) -> Image.Image:
    header = 38
    rendered = []
    for array, label in zip(images, labels):
        image = Image.fromarray(array)
        height = max(1, round(image.height * width / image.width))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height + header), "white")
        canvas.paste(image, (0, header))
        ImageDraw.Draw(canvas).text((8, 10), label, fill="black", font=ImageFont.load_default())
        rendered.append(canvas)
    total = Image.new("RGB", (width * len(rendered), max(x.height for x in rendered)), "white")
    for index, image in enumerate(rendered):
        total.paste(image, (index * width, 0))
    return total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    overview_rows = []
    audit = []
    for case_id in IDS:
        gray = load_gray(ROOT / "data" / "raw" / "TSRS_RSNA-Epiphysis" / "val" / f"{case_id}.jpg")
        shape = gray.shape
        gt = load_mask(ROOT / "data" / "raw" / "TSRS_RSNA-Epiphysis" / "val_labels" / f"{case_id}.png", shape)
        r317 = load_mask(ROOT / "outputs" / "nnunet" / "r317_image_centers_only" / "masks_selected" / f"{case_id}.png", shape)
        r344 = load_mask(OUT / "r344_masks" / f"{case_id}.png", shape)
        diff = r317 ^ r344
        box = crop_box(diff)
        panels = [
            np.repeat(gray[..., None], 3, axis=2),
            overlay(gray, gt, (30, 220, 70)),
            overlay(gray, r317, (0, 210, 255)),
            overlay(gray, r344, (255, 70, 220)),
            difference_panel(gray, r317, r344),
        ]
        labels = ["Original", "GT (green)", "R317 (cyan)", "R344 (magenta)", "Diff: red removed / yellow added"]
        full = tile(panels, labels, width=320)
        full.save(OUT / f"{case_id}_full.png")
        x0, y0, x1, y1 = box
        zoom = tile([p[y0:y1, x0:x1] for p in panels], labels, width=320)
        zoom.save(OUT / f"{case_id}_zoom.png")
        overview_rows.append(np.asarray(zoom.resize((1200, round(zoom.height * 1200 / zoom.width)))))
        labels_cc, count = ndimage.label(diff, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = sorted(np.bincount(labels_cc.reshape(-1))[1:].tolist(), reverse=True)
        audit.append({
            "image": f"{case_id}.png",
            "changed_pixels": int(diff.sum()),
            "removed_pixels": int((r317 & ~r344).sum()),
            "added_pixels": int((r344 & ~r317).sum()),
            "changed_components": int(count),
            "largest_changed_components": sizes[:5],
            "zoom_box_xyxy": list(box),
        })
    max_width = max(row.shape[1] for row in overview_rows)
    overview = Image.new("RGB", (max_width, sum(row.shape[0] for row in overview_rows)), "white")
    y = 0
    for row in overview_rows:
        overview.paste(Image.fromarray(row), (0, y)); y += row.shape[0]
    overview.save(OUT / "r344_outlier_overview.png")
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
