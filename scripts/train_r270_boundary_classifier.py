"""R270 train-only boundary classifier pilot.

The candidate CSVs are generated without reading labels.  Labels in the train
CSV are used only to fit the classifier; the val CSV is an untouched
original-val audit.  No masks are edited and clean-test-v2 is rejected.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


FEATURES = ["neck_score", "saddle_score", "gray_valley_score", "two_side_support",
            "basin_balance", "score", "peak_min"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_xy(rows: list[dict[str, str]], positive_floor: float, negative_ceil: float,
            destructive_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, keep = [], [], []
    for row in rows:
        inter = float(row.get("seam_gt_interinstance_fraction", 0.0))
        destructive = float(row.get("seam_destructive_fg_fraction", 0.0))
        # A positive is a candidate near a true instance contact without
        # mostly cutting unrelated foreground. Ambiguous rows are excluded.
        if inter >= positive_floor and destructive <= destructive_floor:
            label = 1
        elif inter <= negative_ceil or destructive >= 0.80:
            label = 0
        else:
            continue
        vals = [float(row.get(k, 0.0)) for k in FEATURES]
        if not np.all(np.isfinite(vals)):
            continue
        x.append(vals); y.append(label); keep.append(1)
    return np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(keep, np.uint8)


class BoundaryMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 24), nn.ReLU(), nn.Dropout(0.10),
                                 nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def metrics(prob: np.ndarray, y: np.ndarray) -> dict:
    result = {"n": int(len(y)), "positive": int(y.sum())}
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        pred = prob >= threshold
        tp = int((pred & (y > 0.5)).sum()); fp = int((pred & (y < 0.5)).sum())
        fn = int((~pred & (y > 0.5)).sum())
        result[f"t{threshold:.2f}_precision"] = float(tp / max(1, tp + fp))
        result[f"t{threshold:.2f}_recall"] = float(tp / max(1, tp + fn))
        result[f"t{threshold:.2f}_selected"] = int(pred.sum())
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, default=Path("outputs/analysis/r269_wrist_internal_merge_roi_train/candidates.csv"))
    p.add_argument("--val-csv", type=Path, default=Path("outputs/analysis/r269_wrist_internal_merge_roi_fullval/candidates.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/r270_boundary_classifier_pilot"))
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--positive-floor", type=float, default=0.05)
    p.add_argument("--negative-ceil", type=float, default=0.01)
    p.add_argument("--destructive-floor", type=float, default=0.50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if "clean-test" in " ".join(map(str, vars(args).values())).lower():
        raise RuntimeError("R270 forbids clean-test-v2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = load_rows(args.train_csv), load_rows(args.val_csv)
    xtr, ytr, _ = make_xy(train_rows, args.positive_floor, args.negative_ceil, args.destructive_floor)
    xva, yva, _ = make_xy(val_rows, args.positive_floor, args.negative_ceil, args.destructive_floor)
    if len(xtr) < 10 or len(np.unique(ytr)) < 2:
        raise RuntimeError(f"insufficient train labels: n={len(xtr)} classes={np.unique(ytr).tolist()}")
    mean, std = xtr.mean(0), xtr.std(0); std[std < 1e-5] = 1.0
    device = torch.device(args.device)
    model = BoundaryMLP(xtr.shape[1]).to(device)
    tx = torch.from_numpy((xtr - mean) / std).to(device)
    ty = torch.from_numpy(ytr).to(device)
    pos = max(1.0, float(ytr.sum())); neg = max(1.0, float(len(ytr) - ytr.sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    history = []
    for epoch in range(args.epochs):
        model.train(); opt.zero_grad(); loss = criterion(model(tx), ty); loss.backward(); opt.step()
        if epoch == 0 or (epoch + 1) % 25 == 0:
            history.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        ptr = torch.sigmoid(model(torch.from_numpy((xtr - mean) / std).to(device))).cpu().numpy()
        pva = torch.sigmoid(model(torch.from_numpy((xva - mean) / std).to(device))).cpu().numpy()
    result = {"run_id": "R270_boundary_classifier_pilot", "train_source": str(args.train_csv),
              "val_source": str(args.val_csv), "clean_test_used": False,
              "features": FEATURES, "device": str(device), "epochs": args.epochs,
              "label_rule": {"positive_interinstance_ge": args.positive_floor,
                              "negative_interinstance_le": args.negative_ceil,
                              "positive_destructive_le": args.destructive_floor},
              "train": metrics(ptr, ytr), "original_val": metrics(pva, yva),
              "history": history, "normalization_mean": mean.tolist(),
              "normalization_std": std.tolist()}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "features": FEATURES, "label_rule": result["label_rule"]}, args.output_dir / "boundary_classifier.pt")
    print(json.dumps({k: result[k] for k in ("run_id", "device", "train", "original_val", "label_rule")}, indent=2))


if __name__ == "__main__":
    main()
