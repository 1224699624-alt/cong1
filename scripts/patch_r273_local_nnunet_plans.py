"""Patch generated 2D plans to fit the local 6 GB GPU."""
from __future__ import annotations
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser(); p.add_argument("plan", type=Path); p.add_argument("--patch", type=int, default=512); p.add_argument("--batch", type=int, default=2); a = p.parse_args()
data = json.loads(a.plan.read_text(encoding="utf-8"))
cfg = data["configurations"]["2d"]
cfg["patch_size"] = [a.patch, a.patch]
cfg["batch_size"] = a.batch
a.plan.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(json.dumps({"plan": str(a.plan), "patch_size": cfg["patch_size"], "batch_size": cfg["batch_size"]}, indent=2))
