#!/usr/bin/env python3
"""Create paired qualitative panels for the R326 RAM validation experiment.

The script ranks validation cases by per-image overlap-Dice improvement of the
layer-projection arm over the frozen baseline. It never loads the RAM test set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior  # noqa: E402
from train_r326_ram_nnunet_layer_projection_prior import (  # noqa: E402
    LayerProjectionAdapter,
    WristLayerDataset,
)


PALETTE = np.asarray([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
    [170, 110, 40], [255, 250, 200],
], dtype=np.uint8)


def overlap_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = prediction.sum(0) >= 2
    true = target.sum(0) >= 2
    return float((2 * np.logical_and(pred, true).sum() + 1) / (pred.sum() + true.sum() + 1))


def color_instances(mask: np.ndarray, gray: np.ndarray) -> np.ndarray:
    result = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    membership = mask.sum(0)
    for channel in range(mask.shape[0]):
        active = mask[channel]
        result[active] = 0.30 * result[active] + 0.70 * PALETTE[channel]
    # Make true multi-membership pixels explicit instead of hiding them behind
    # whichever instance happens to be painted last.
    result[membership >= 2] = [255, 0, 255]
    return np.clip(result, 0, 255).astype(np.uint8)


def overlap_error(prediction: np.ndarray, target: np.ndarray, gray: np.ndarray) -> np.ndarray:
    pred = prediction.sum(0) >= 2
    true = target.sum(0) >= 2
    result = np.repeat(gray[..., None], 3, axis=2)
    result[np.logical_and(true, pred)] = [0, 220, 0]
    result[np.logical_and(true, ~pred)] = [255, 0, 0]
    result[np.logical_and(~true, pred)] = [255, 220, 0]
    return result


def add_title(array: np.ndarray, title: str, height: int = 28) -> np.ndarray:
    canvas = Image.new("RGB", (array.shape[1], array.shape[0] + height), "white")
    canvas.paste(Image.fromarray(array), (0, height))
    ImageDraw.Draw(canvas).text((7, 7), title, fill="black", font=ImageFont.load_default())
    return np.asarray(canvas)


def crop_box(target: np.ndarray, predictions: list[np.ndarray], margin: int = 35) -> tuple[int, int, int, int]:
    interest = target.sum(0) >= 2
    for prediction in predictions:
        interest |= prediction.sum(0) >= 2
    ys, xs = np.where(interest)
    height, width = interest.shape
    if not len(xs):
        return (0, 0, width, height)
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, width)
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, height)
    side = max(x1 - x0, y1 - y0, 96)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, x1 = max(cx - side // 2, 0), min(cx + (side + 1) // 2, width)
    y0, y1 = max(cy - side // 2, 0), min(cy + (side + 1) // 2, height)
    return (x0, y0, x1, y1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"G:\gutou\RAM-W600"))
    parser.add_argument("--baseline-checkpoint", type=Path,
                        default=REPO / "outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth")
    parser.add_argument("--experiment", type=Path,
                        default=REPO / "outputs/ram_w600/r326_layer_projection_prior_batched")
    parser.add_argument("--output", type=Path,
                        default=REPO / "outputs/visualizations/r326_layer_projection_prior")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--num-cases", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = NnUNetMultiLabelPrior().to(device)
    baseline.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device,
                                        weights_only=False)["model"], strict=True)
    baseline.eval()
    adapters = {}
    for key, filename in (("seg_only", "seg_only_adapter_best.pth"),
                          ("projection", "layer_projection_adapter_best.pth")):
        adapter = LayerProjectionAdapter().to(device)
        payload = torch.load(args.experiment / filename, map_location=device, weights_only=False)
        adapter.load_state_dict(payload["adapter"], strict=True)
        adapter.eval()
        adapters[key] = adapter

    dataset = WristLayerDataset(args.dataset_root, "val", args.size, False)
    records = []
    with torch.inference_mode():
        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(device)
            target = (sample["mask"].numpy() > 0.5)
            base_logits, _ = baseline(image, False)
            predictions = {"baseline": (torch.sigmoid(base_logits)[0].cpu().numpy() >= 0.5)}
            for key, adapter in adapters.items():
                logits, _ = adapter(image, base_logits)
                predictions[key] = (torch.sigmoid(logits)[0].cpu().numpy() >= 0.5)
            scores = {key: overlap_dice(value, target) for key, value in predictions.items()}
            records.append({"case": sample["case"], "target": target,
                            "gray": (sample["raw"][0].numpy() * 255).astype(np.uint8),
                            "predictions": predictions, "scores": scores,
                            "delta_projection_minus_baseline": scores["projection"] - scores["baseline"],
                            "delta_projection_minus_seg_only": scores["projection"] - scores["seg_only"],
                            "gt_overlap_pixels": int((target.sum(0) >= 2).sum())})

    eligible = [record for record in records if record["gt_overlap_pixels"] > 0]
    eligible.sort(key=lambda record: record["delta_projection_minus_baseline"], reverse=True)
    selected = eligible[:args.num_cases]
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for record in selected:
        target, gray, predictions = record["target"], record["gray"], record["predictions"]
        panels = [
            add_title(np.repeat(gray[..., None], 3, axis=2), "Radiograph"),
            add_title(color_instances(target, gray), "GT (magenta=overlap)"),
            add_title(color_instances(predictions["baseline"], gray),
                      f"Frozen nnU-Net  OD={record['scores']['baseline']:.4f}"),
            add_title(color_instances(predictions["seg_only"], gray),
                      f"Seg-only adapter  OD={record['scores']['seg_only']:.4f}"),
            add_title(color_instances(predictions["projection"], gray),
                      f"Layer prior  OD={record['scores']['projection']:.4f}"),
            add_title(overlap_error(predictions["projection"], target, gray),
                      "Layer error: TP green / FN red / FP yellow"),
        ]
        full = np.concatenate(panels, axis=1)
        Image.fromarray(full).save(args.output / f"{record['case']}_full.png")

        x0, y0, x1, y1 = crop_box(target, list(predictions.values()))
        zoom_panels = []
        for panel in panels:
            title_height = panel.shape[0] - args.size
            crop = panel[title_height + y0:title_height + y1, x0:x1]
            crop = np.asarray(Image.fromarray(crop).resize((256, 256), Image.Resampling.NEAREST))
            zoom_panels.append(crop)
        Image.fromarray(np.concatenate(zoom_panels, axis=1)).save(
            args.output / f"{record['case']}_overlap_zoom.png")
        report.append({key: value for key, value in record.items()
                       if key not in {"target", "gray", "predictions"}})

    (args.output / "selection.json").write_text(json.dumps({
        "split": "validation", "test_used": False, "ranking": "projection overlap Dice minus baseline",
        "color_legend": {"magenta": "multi-bone overlap", "green": "overlap TP",
                         "red": "overlap FN", "yellow": "overlap FP"},
        "cases": report,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "output": str(args.output),
                      "selected_cases": [record["case"] for record in selected]}, indent=2))


if __name__ == "__main__":
    main()
