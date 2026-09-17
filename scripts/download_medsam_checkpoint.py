#!/usr/bin/env python3
"""Download the public MedSAM ViT-B checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from tqdm import tqdm


URLS = [
    "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1",
    "https://huggingface.co/wanglab/medsam-vit-base/resolve/main/medsam_vit_b.pth",
]
EXPECTED_MD5 = "3bb6db55bd0c9ca30b61248bca72f8d6"


class DownloadProgress:
    def __init__(self) -> None:
        self.progress: tqdm | None = None

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if self.progress is None:
            self.progress = tqdm(total=total_size if total_size > 0 else None, unit="B", unit_scale=True)
        downloaded = block_num * block_size
        self.progress.update(downloaded - self.progress.n)
        if total_size > 0 and downloaded >= total_size:
            self.progress.close()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MedSAM ViT-B checkpoint.")
    parser.add_argument("--output", default="checkpoints/medsam_vit_b.pth")
    parser.add_argument("--skip-md5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        print(f"Checkpoint already exists: {output}")
        if not args.skip_md5:
            current_md5 = file_md5(output)
            if current_md5 != EXPECTED_MD5:
                raise ValueError(f"Existing checkpoint md5 mismatch: {current_md5}")
        return

    last_error: Exception | None = None
    for url in URLS:
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, output, DownloadProgress())
            if not args.skip_md5:
                current_md5 = file_md5(output)
                if current_md5 != EXPECTED_MD5:
                    output.unlink(missing_ok=True)
                    raise ValueError(f"Downloaded checkpoint md5 mismatch: {current_md5}")
            print(f"Saved checkpoint: {output}")
            return
        except Exception as exc:
            last_error = exc
            print(f"Download failed: {exc}")

    raise RuntimeError(f"All checkpoint download URLs failed. Last error: {last_error}")


if __name__ == "__main__":
    main()
