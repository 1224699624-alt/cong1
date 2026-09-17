#!/usr/bin/env python3
"""Create metric tables and real RAM validation comparisons for R324 lambda sweep."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from train_r284_ramw600_overlap_prior import BONE_NAMES, NnUNetMultiLabelPrior, WristDataset
from train_r324_ram_nnunet_explicit_overlap_iem import surfaces


PALETTE = np.asarray([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
    [170, 110, 40], [255, 250, 200],
], dtype=np.uint8)


def load_model(checkpoint: Path, device: torch.device) -> NnUNetMultiLabelPrior:
    model = NnUNetMultiLabelPrior().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"], strict=True)
    model.eval()
    return model


def case_overlap_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    po, to = prediction.sum(0) >= 2, target.sum(0) >= 2
    inter = np.logical_and(po, to).sum()
    dice = float((2 * inter + 1) / (po.sum() + to.sum() + 1))
    iou = float((inter + 1) / (po.sum() + to.sum() - inter + 1))
    nsd, msd, failed = surfaces(po, to)
    return {"dsc": dice, "iou": iou, "nsd": nsd,
            "msd": msd if np.isfinite(msd) else 999.0, "failed": float(failed)}


def instance_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape[1:], 3), dtype=np.uint8)
    for channel, color in enumerate(PALETTE):
        rgb[mask[channel]] = color
    overlap = mask.sum(0) >= 2
    rgb[overlap] = [255, 255, 255]
    return rgb


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    color = instance_rgb(mask)
    base = np.repeat(image[..., None], 3, axis=2)
    foreground = mask.any(0)
    out = base.copy()
    out[foreground] = ((1 - alpha) * base[foreground] + alpha * color[foreground]).astype(np.uint8)
    return out


def overlap_error(image: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    po, to = prediction.sum(0) >= 2, target.sum(0) >= 2
    base = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    out = base.copy()
    for region, color in ((po & to, [0, 220, 0]), (~po & to, [255, 30, 30]), (po & ~to, [255, 215, 0])):
        out[region] = 0.35 * out[region] + 0.65 * np.asarray(color)
    return out.astype(np.uint8)


def roi_from_target(target: np.ndarray, margin: int = 18) -> tuple[int, int, int, int]:
    overlap = target.sum(0) >= 2
    ys, xs = np.where(overlap)
    if not len(xs): return 0, 0, target.shape[2], target.shape[1]
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, target.shape[2])
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, target.shape[1])
    return x0, y0, x1, y1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--result", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = WristDataset(args.dataset_root, "val", args.size, False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    labels = [x[0] for x in args.checkpoint]
    checkpoints = {label: Path(path) for label, path in args.checkpoint}

    metrics: dict[str, dict[str, dict[str, float]]] = {label: {} for label in labels}
    targets: dict[str, np.ndarray] = {}
    for label in labels:
        model = load_model(checkpoints[label], device)
        with torch.no_grad():
            for batch in loader:
                case = str(batch["case"][0])
                target = batch["mask"][0].numpy() > 0.5
                logits, _ = model(batch["image"].to(device), False)
                pred = (torch.sigmoid(logits)[0].cpu().numpy() >= 0.5)
                metrics[label][case] = case_overlap_metrics(pred, target)
                targets[case] = target
        del model
        torch.cuda.empty_cache()

    base_label = labels[0]
    preferred = "0.002" if "0.002" in labels else labels[1]
    strong = "0.010" if "0.010" in labels else labels[-1]
    cases = sorted(targets)
    improvement = sorted(cases, key=lambda c: metrics[preferred][c]["dsc"] - metrics[base_label][c]["dsc"], reverse=True)
    degradation = sorted(cases, key=lambda c: metrics[strong][c]["dsc"] - metrics[base_label][c]["dsc"])
    selected: list[str] = []
    for case in improvement[: max(args.max_cases // 2, 1)] + degradation[: max(args.max_cases // 2, 1)]:
        if case not in selected: selected.append(case)
    selected = selected[:args.max_cases]

    predictions: dict[str, dict[str, np.ndarray]] = {label: {} for label in labels}
    selected_set = set(selected)
    for label in labels:
        model = load_model(checkpoints[label], device)
        with torch.no_grad():
            for batch in loader:
                case = str(batch["case"][0])
                if case not in selected_set: continue
                logits, _ = model(batch["image"].to(device), False)
                predictions[label][case] = torch.sigmoid(logits)[0].cpu().numpy() >= 0.5
        del model
        torch.cuda.empty_cache()

    with (args.output / "case_overlap_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "lambda", "overlap_dsc", "overlap_iou", "overlap_nsd_2px", "overlap_msd_px"])
        for case in cases:
            for label in labels:
                row = metrics[label][case]
                writer.writerow([case, label, row["dsc"], row["iou"], row["nsd"], row["msd"]])

    result_rows = []
    for label, path in args.result:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        value = payload["plain_val"] if label == "0" else payload.get("prior_val", payload.get("explicit_overlap_iem_val"))
        result_rows.append({"lambda": float(label), **{k: value[k] for k in (
            "macro_dsc", "macro_iou", "overlap_dsc", "overlap_iou", "overlap_nsd_2px", "overlap_msd_px")}})
    result_rows.sort(key=lambda r: r["lambda"])
    with (args.output / "lambda_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader(); writer.writerows(result_rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    x = [r["lambda"] for r in result_rows]
    for ax, key, title, direction in (
        (axes[0, 0], "macro_dsc", "Macro DSC", "higher is better"),
        (axes[0, 1], "overlap_dsc", "Overlap DSC", "higher is better"),
        (axes[1, 0], "overlap_nsd_2px", "Overlap NSD @ 2 px", "higher is better"),
        (axes[1, 1], "overlap_msd_px", "Overlap MSD (px)", "lower is better"),
    ):
        y = [r[key] for r in result_rows]
        ax.plot(x, y, marker="o", linewidth=2)
        for xx, yy in zip(x, y): ax.annotate(f"{yy:.6f}", (xx, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(f"{title} ({direction})"); ax.set_xlabel("lambda"); ax.grid(alpha=.25)
    fig.savefig(args.output / "lambda_metric_curves.png", dpi=180)
    plt.close(fig)

    display_labels = [base_label, preferred, "0.006" if "0.006" in labels else labels[-2], strong]
    for rank, case in enumerate(selected, 1):
        raw = np.asarray(Image.open(args.dataset_root / "BoneSegmentation" / "images" / f"{case}.bmp").convert("L").resize((args.size, args.size), Image.Resampling.BILINEAR))
        target = targets[case]
        preds = {label: predictions[label][case] for label in display_labels}
        if case.endswith("_R"):
            target = np.flip(target, axis=2).copy()
            preds = {k: np.flip(v, axis=2).copy() for k, v in preds.items()}
        x0, y0, x1, y1 = roi_from_target(target)
        columns = ["Image", "GT"] + [f"lambda={x}" if x != "0" else "Plain" for x in display_labels]
        fig, axes = plt.subplots(2, len(columns), figsize=(3 * len(columns), 6), constrained_layout=True)
        top = [np.repeat(raw[..., None], 3, 2), overlay(raw, target)] + [overlay(raw, preds[x]) for x in display_labels]
        bottom = [np.repeat(raw[..., None], 3, 2), overlap_error(raw, target, target)] + [overlap_error(raw, preds[x], target) for x in display_labels]
        for c, (title, image) in enumerate(zip(columns, top)):
            axes[0, c].imshow(image); axes[0, c].set_title(title); axes[0, c].axis("off")
        for c, image in enumerate(bottom):
            axes[1, c].imshow(image[y0:y1, x0:x1]); axes[1, c].axis("off")
            if c >= 2:
                label = display_labels[c - 2]; m = metrics[label][case]
                axes[1, c].set_title(f"ROI DSC {m['dsc']:.4f} | NSD {m['nsd']:.4f}", fontsize=9)
        axes[1, 0].set_title("Overlap ROI"); axes[1, 1].set_title("GT overlap (green)")
        fig.suptitle(f"{case} | overlap error: green TP, red missed, yellow false", fontsize=13)
        fig.savefig(args.output / f"case_{rank:02d}_{case}.png", dpi=150)
        plt.close(fig)

    summary = {"selected_cases": selected, "labels": labels, "display_labels": display_labels,
               "selection": f"top {args.max_cases//2} {preferred} improvements and top {args.max_cases//2} {strong} degradations vs plain",
               "test_used": False}
    (args.output / "visualization_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
