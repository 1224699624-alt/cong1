#!/usr/bin/env python3
"""Cleanly replace a corrupted TransUNet NPZ and resume the R320 queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import paramiko
import requests


URL = "https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"
EXPECTED_SIZE = 461_217_452


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_npz(path: Path) -> dict[str, object]:
    if path.stat().st_size != EXPECTED_SIZE:
        raise RuntimeError(f"unexpected size: {path.stat().st_size}/{EXPECTED_SIZE}")
    with zipfile.ZipFile(path) as archive:
        entries = len(archive.infolist())
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"ZIP CRC failure: {bad_member}")
    with np.load(path, allow_pickle=False) as weights:
        keys = list(weights.files)
        for key in keys:
            array = weights[key]
            _ = (array.shape, array.dtype)
    return {"size": path.stat().st_size, "entries": entries, "arrays": len(keys), "sha256": sha256(path)}


def download_clean(target: Path) -> dict[str, object]:
    partial = target.with_suffix(target.suffix + ".download.part")
    if partial.exists():
        partial.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"clean download: {URL} -> {partial}", flush=True)
    with requests.get(URL, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        content_length = int(response.headers.get("content-length", 0))
        if content_length and content_length != EXPECTED_SIZE:
            raise RuntimeError(f"server content-length mismatch: {content_length}/{EXPECTED_SIZE}")
        written = 0
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                print(f"downloaded {written}/{EXPECTED_SIZE}", flush=True)
    if written != EXPECTED_SIZE:
        raise RuntimeError(f"download size mismatch: {written}/{EXPECTED_SIZE}")
    result = validate_npz(partial)
    os.replace(partial, target)
    print(f"local validation passed: {json.dumps(result, sort_keys=True)}", flush=True)
    return result


def upload_and_replace(args: argparse.Namespace, local: Path, local_info: dict[str, object]) -> dict[str, object]:
    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(f"missing environment variable: {args.password_env}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        port=args.port,
        username=args.user,
        password=password,
        timeout=30,
    )
    remote_final = f"{args.remote_checkpoints}/R50_ViT-B_16.npz"
    remote_partial = remote_final + ".clean.upload.part"
    sftp = client.open_sftp()
    print(f"uploading clean file -> {remote_partial}", flush=True)
    with local.open("rb") as source, sftp.open(remote_partial, "wb") as target:
        transferred = 0
        while True:
            chunk = source.read(4 * 1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            transferred += len(chunk)
            print(f"uploaded {transferred}/{EXPECTED_SIZE}", flush=True)
        target.flush()
    if sftp.stat(remote_partial).st_size != EXPECTED_SIZE:
        raise RuntimeError("remote upload size mismatch")
    sftp.close()

    validation = f"""REMOTE_PARTIAL={json.dumps(remote_partial)} REMOTE_FINAL={json.dumps(remote_final)} /root/miniconda3/bin/python - <<'PY'
import hashlib, json, os, zipfile
import numpy as np
from pathlib import Path
p=Path(os.environ['REMOTE_PARTIAL'])
final=Path(os.environ['REMOTE_FINAL'])
expected={EXPECTED_SIZE}
if p.stat().st_size != expected:
    raise SystemExit(f'size mismatch: {{p.stat().st_size}}/{{expected}}')
with zipfile.ZipFile(p) as z:
    bad=z.testzip()
    entries=len(z.infolist())
if bad is not None:
    raise SystemExit(f'bad ZIP member: {{bad}}')
with np.load(p, allow_pickle=False) as data:
    keys=list(data.files)
    for key in keys:
        arr=data[key]
        _=(arr.shape,arr.dtype)
h=hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda:f.read(4*1024*1024),b''):
        h.update(chunk)
digest=h.hexdigest()
if digest != {json.dumps(local_info['sha256'])}:
    raise SystemExit(f'SHA mismatch: {{digest}}')
if final.exists():
    corrupt=final.with_name(final.name+'.corrupt-'+str(int(__import__('time').time())))
    os.replace(final,corrupt)
os.replace(p,final)
print(json.dumps({{'size':expected,'entries':entries,'arrays':len(keys),'sha256':digest,'final':str(final)}}))
PY"""
    _, stdout, stderr = client.exec_command(validation, timeout=600)
    output = stdout.read().decode("utf-8", "replace").strip()
    error = stderr.read().decode("utf-8", "replace").strip()
    code = stdout.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(f"remote validation failed ({code}): {error or output}")
    print(f"remote validation passed: {output}", flush=True)
    remote_info = json.loads(output.splitlines()[-1])

    if args.restart_queue:
        command = f"""cd {args.remote_root} && \
screen -S r320_multibackbone_queue -X quit >/dev/null 2>&1 || true; \
screen -dmS r320_multibackbone_queue bash -lc './run_r320_multibackbone_queue.sh 2>&1 | tee outputs/bridge_logs/r320_multibackbone_queue_repaired.log'; \
screen -ls"""
        _, stdout, stderr = client.exec_command(command, timeout=30)
        queue_output = stdout.read().decode("utf-8", "replace")
        queue_error = stderr.read().decode("utf-8", "replace")
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(queue_error or queue_output)
        print(queue_output, flush=True)
    client.close()
    return remote_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    parser.add_argument("--remote-root", default="/root/autodl-tmp/YOLO_SAM_generic_src")
    parser.add_argument("--remote-checkpoints", default="/root/autodl-tmp/external/checkpoints")
    parser.add_argument("--local", type=Path, default=Path("checkpoints/R50_ViT-B_16.npz.clean"))
    parser.add_argument("--restart-queue", action="store_true")
    args = parser.parse_args()
    started = time.time()
    local_info = download_clean(args.local)
    remote_info = upload_and_replace(args, args.local, local_info)
    result = {
        "status": "complete",
        "local": local_info,
        "remote": remote_info,
        "elapsed_seconds": time.time() - started,
        "queue_restarted": args.restart_queue,
    }
    output = Path("outputs/analysis/r320_transunet_weight_repair.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"repair complete: {output}", flush=True)


if __name__ == "__main__":
    main()
