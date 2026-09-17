#!/usr/bin/env python3
"""Upload one file over password-authenticated SFTP using an environment secret."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path, PurePosixPath

import paramiko


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--remote", type=PurePosixPath, required=True)
    parser.add_argument("--password-env", default="SSH_PASSWORD")
    args = parser.parse_args()
    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(f"missing password environment variable: {args.password_env}")
    if not args.local.is_file():
        raise FileNotFoundError(args.local)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, port=args.port, username=args.user, password=password, timeout=30)
    sftp = client.open_sftp()
    remote = str(args.remote)
    temporary = remote + ".part"
    local_size = args.local.stat().st_size
    try:
        remote_size = sftp.stat(remote).st_size
    except OSError:
        remote_size = -1
    if remote_size == local_size:
        print(f"already complete bytes={remote_size}", flush=True)
        sftp.close()
        client.close()
        return

    try:
        partial_size = sftp.stat(temporary).st_size
    except OSError:
        partial_size = 0
    if partial_size > local_size:
        sftp.remove(temporary)
        partial_size = 0

    print(
        f"uploading {args.local} -> {temporary} from byte {partial_size}/{local_size}",
        flush=True,
    )
    with args.local.open("rb") as source, sftp.open(temporary, "ab") as target:
        source.seek(partial_size)
        transferred = partial_size
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            transferred += len(chunk)
            if transferred == local_size or transferred // (32 * 1024 * 1024) != (transferred - len(chunk)) // (32 * 1024 * 1024):
                print(f"uploaded {transferred}/{local_size}", file=sys.stdout, flush=True)
        target.flush()

    uploaded_size = sftp.stat(temporary).st_size
    if uploaded_size != local_size:
        raise RuntimeError(f"partial size mismatch: local={local_size}, remote={uploaded_size}")
    sftp.rename(temporary, remote)
    remote_size = sftp.stat(remote).st_size
    sftp.close()
    client.close()
    if local_size != remote_size:
        raise RuntimeError(f"size mismatch: local={local_size}, remote={remote_size}")
    print(f"completed bytes={remote_size}", flush=True)


if __name__ == "__main__":
    main()
