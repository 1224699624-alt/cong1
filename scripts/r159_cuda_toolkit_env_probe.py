#!/usr/bin/env python3
"""Probe whether an isolated CUDA-toolkit-capable env is feasible for MaskDINO/Mask2Former."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], timeout: int = 20) -> dict[str, object]:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"command": command, "error": repr(exc)}


def path_exists(paths: list[str]) -> dict[str, bool]:
    return {path: Path(path).exists() for path in paths}


def main() -> None:
    candidates = {
        "conda": shutil.which("conda"),
        "mamba": shutil.which("mamba"),
        "micromamba": shutil.which("micromamba"),
        "nvcc": shutil.which("nvcc"),
        "gcc": shutil.which("gcc"),
        "g++": shutil.which("g++"),
        "pip": shutil.which("pip"),
    }
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-11.8",
        "/usr/local/cuda-12",
        "/usr/local/cuda-12.1",
        "/usr/local/cuda-12.4",
        "/opt/cuda",
        str(Path.home() / ".conda"),
        str(Path.home() / "miniconda3"),
        str(Path.home() / "anaconda3"),
    ]
    py = "/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"
    report: dict[str, object] = {
        "cwd": os.getcwd(),
        "env_path": py,
        "which": candidates,
        "path_exists": path_exists(cuda_paths),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]),
        "python_torch": run([
            py,
            "-c",
            "import json, torch; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()}))",
        ]),
        "python_prefix": run([py, "-c", "import sys,sysconfig,json; print(json.dumps({'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'include': sysconfig.get_paths().get('include')}))"]),
        "gcc": run(["gcc", "--version"]),
        "gxx": run(["g++", "--version"]),
        "nvcc_version": run(["nvcc", "--version"]),
        "conda_info": run(["conda", "info", "--json"]),
        "conda_search_cuda_toolkit": run(["conda", "search", "-c", "nvidia", "cuda-toolkit=11.8", "--json"], timeout=60),
        "pip_detectron2_dry_query": run([
            py,
            "-m",
            "pip",
            "index",
            "versions",
            "detectron2",
        ], timeout=60),
        "pip_fvcore_dry_query": run([
            py,
            "-m",
            "pip",
            "index",
            "versions",
            "fvcore",
        ], timeout=60),
        "disk": run(["df", "-h", ".", str(Path.home())]),
    }
    out = Path("outputs/analysis/r159_cuda_toolkit_env_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
