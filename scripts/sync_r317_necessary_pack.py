#!/usr/bin/env python3
"""Synchronize the non-redundant R317 reproducibility pack over SFTP."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default=os.environ.get("R317_SYNC_PASSWORD"))
    parser.add_argument("--remote-root", default="/root/autodl-tmp/YOLO_SAM_generic_src")
    parser.add_argument("--local-root", type=Path, default=Path(r"G:\gutou\YOLO+SAM"))
    args = parser.parse_args()
    if not args.password:
        raise RuntimeError("Provide --password or R317_SYNC_PASSWORD")
    root = args.remote_root
    fold = ("outputs/nnunet/r317_image_centers_only/nnUNet_results/"
            "Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/"
            "nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0")
    model = str(Path(fold).parent).replace("\\", "/")
    data = "outputs/nnunet/r317_image_centers_only/data"
    find_parts = [
        "find outputs/pair_prior/r317_image_centers_only -type f",
        "find outputs/priors/r317_image_centers_only -type f",
        "find outputs/analysis -maxdepth 1 -type f -name 'r317_image_centers_only*'",
        "find outputs/visualizations/r317_image_centers_only_vs_r260 -type f",
        "find outputs/bridge_logs -maxdepth 1 -type f -name 'r317_image_centers_only*'",
        (f"printf '%s\\n' outputs/nnunet/r317_image_centers_only/r202_two_channel_zero_prior.pth "
         f"{fold}/checkpoint_r201_selected.pth"),
        f"find {fold} -maxdepth 1 -type f ! -name '*.pth'",
        f"find {fold}/validation -type f",
        "find outputs/nnunet/r317_image_centers_only/masks_selected -type f",
        f"find {model} -maxdepth 1 -type f",
        f"find {data} -maxdepth 1 -type f",
        (f"find {data}/nnUNet_preprocessed/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D "
         "-maxdepth 1 -type f"),
    ]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, port=args.port, username=args.user,
                   password=args.password, timeout=30)
    command = f"cd '{root}' && {{ " + "; ".join(find_parts) + "; } | sort -u"
    _, stdout, stderr = client.exec_command(command, timeout=120)
    paths = [x.strip() for x in stdout.read().decode().splitlines() if x.strip()]
    error = stderr.read().decode()
    if error:
        print(error, file=sys.stderr)
    if not paths:
        raise RuntimeError("No remote files selected")
    quoted = " ".join("'" + x.replace("'", "'\\''") + "'" for x in paths)
    _, stdout, stderr = client.exec_command(
        f"cd '{root}' && sha256sum {quoted}", timeout=600)
    remote_hash = {
        line.split(None, 1)[1].lstrip("*"): line.split(None, 1)[0]
        for line in stdout.read().decode().splitlines()
        if len(line.split(None, 1)) == 2
    }
    error = stderr.read().decode()
    if error:
        print(error, file=sys.stderr)
    if len(remote_hash) != len(paths):
        raise RuntimeError(f"Remote hash coverage {len(remote_hash)}/{len(paths)}")
    sftp = client.open_sftp()
    sizes = {path: sftp.stat(f"{root}/{path}").st_size for path in paths}
    rows, downloaded, skipped = [], 0, 0
    started = time.time()
    for index, relative in enumerate(paths, 1):
        local = args.local_root / Path(relative)
        local.parent.mkdir(parents=True, exist_ok=True)
        expected, size = remote_hash[relative], sizes[relative]
        current = None
        if local.is_file() and local.stat().st_size == size:
            current = hashlib.sha256(local.read_bytes()).hexdigest()
        if current == expected:
            skipped += size
        else:
            temporary = local.with_suffix(local.suffix + ".transfer_tmp")
            sftp.get(f"{root}/{relative}", str(temporary))
            actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Hash mismatch {relative}: {actual} != {expected}")
            os.replace(temporary, local)
            downloaded += size
        rows.append({"relative_path": relative, "bytes": size, "sha256": expected})
        if index % 100 == 0 or index == len(paths):
            print(json.dumps({
                "files": f"{index}/{len(paths)}",
                "downloaded_mb": round(downloaded / 2**20, 1),
                "verified_existing_mb": round(skipped / 2**20, 1),
                "elapsed_s": round(time.time() - started, 1),
            }), flush=True)
    sftp.close()
    client.close()
    manifest = (args.local_root / "outputs/remote_sync/westd_27945_20260731/"
                "NECESSARY_REPRO_PACK.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "server": f"{args.host}:{args.port}",
        "remote_root": root,
        "selection": ("R317 necessary reproducibility pack; redundant periodic nnU-Net "
                      "checkpoints and regenerable preprocessing caches excluded"),
        "files": rows,
        "total_bytes": sum(row["bytes"] for row in rows),
        "all_sha256_verified": True,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "complete": True,
        "files": len(rows),
        "total_gib": round(sum(sizes.values()) / 2**30, 3),
        "downloaded_gib": round(downloaded / 2**30, 3),
        "manifest": str(manifest),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
