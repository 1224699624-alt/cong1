#!/usr/bin/env python3
"""Sync R176 clean-test-v2 prediction masks from the remote experiment server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import paramiko


DEFAULT_EXPERIMENTS = [
    "r090_online_patch_disagreement_arbitrator",
    "r100_r095b_r097_patch_basic",
    "r110_r100_r108_patch_basic",
    "r130_dinov3_bridge_suppressed_instance_sep",
    "r143_highres_medical_recipe_unet",
    "r165_filtered_reannotated_dinov3_instance_sep",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync R176 clean-test-v2 masks.")
    parser.add_argument("--host", default="10.1.115.157")
    parser.add_argument("--user", default="shenzeyu")
    parser.add_argument("--password", default=None)
    parser.add_argument("--remote-root", default="/home/shenzeyu/workspace/YOLO_SAM_generic_src")
    parser.add_argument("--local-root", default="outputs/ablations_variants")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--split", action="append", default=[], help="Split to sync. Can be repeated. Defaults to test.")
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--sync-labels", action="store_true", help="Also sync labels for local metric audit.")
    parser.add_argument("--remote-raw-root", default="data/raw_variants")
    parser.add_argument("--local-raw-root", default="data/raw_variants")
    parser.add_argument("--summary", default="outputs/analysis/r176_mask_sync_summary.json")
    return parser.parse_args()


def sftp_listdir(sftp: paramiko.SFTPClient, path: str) -> list[str]:
    try:
        return sftp.listdir(path)
    except FileNotFoundError:
        return []


def main() -> None:
    args = parse_args()
    experiments = args.experiment or DEFAULT_EXPERIMENTS
    if not args.password:
        raise SystemExit("Password is required via --password for this local sync helper.")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        args.host,
        username=args.user,
        password=args.password,
        timeout=10,
        banner_timeout=10,
        auth_timeout=10,
    )
    sftp = ssh.open_sftp()

    summary = {
        "dataset": args.dataset,
        "splits": args.split or ["test"],
        "remote_root": args.remote_root,
        "local_root": args.local_root,
        "label_sync": None,
        "experiments": [],
    }

    splits = args.split or ["test"]
    if args.sync_labels:
        label_syncs = []
        for split in splits:
            remote_label_dir = f"{args.remote_root}/{args.remote_raw_root}/{args.dataset}/{split}_labels"
            local_label_dir = Path(args.local_raw_root) / args.dataset / f"{split}_labels"
            local_label_dir.mkdir(parents=True, exist_ok=True)
            label_names = sorted(name for name in sftp_listdir(sftp, remote_label_dir) if name.lower().endswith(".png"))
            copied_labels = 0
            for name in label_names:
                local_path = local_label_dir / name
                if local_path.exists() and local_path.stat().st_size > 0:
                    continue
                sftp.get(f"{remote_label_dir}/{name}", str(local_path))
                copied_labels += 1
            label_syncs.append({
                "split": split,
                "remote_dir": remote_label_dir,
                "local_dir": str(local_label_dir),
                "remote_png_count": len(label_names),
                "copied": copied_labels,
                "local_png_count": len(list(local_label_dir.glob("*.png"))),
            })
        summary["label_sync"] = label_syncs

    for exp in experiments:
        for split in splits:
            remote_dir = f"{args.remote_root}/outputs/ablations_variants/{exp}/{args.dataset}/{split}/masks"
            local_dir = Path(args.local_root) / exp / args.dataset / split / "masks"
            local_dir.mkdir(parents=True, exist_ok=True)
            names = sorted(name for name in sftp_listdir(sftp, remote_dir) if name.lower().endswith(".png"))
            copied = 0
            for name in names:
                local_path = local_dir / name
                if local_path.exists() and local_path.stat().st_size > 0:
                    continue
                sftp.get(f"{remote_dir}/{name}", str(local_path))
                copied += 1
            summary["experiments"].append({
                "experiment": exp,
                "split": split,
                "remote_dir": remote_dir,
                "local_dir": str(local_dir),
                "remote_png_count": len(names),
                "copied": copied,
                "local_png_count": len(list(local_dir.glob("*.png"))),
            })

    sftp.close()
    ssh.close()

    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
