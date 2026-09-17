#!/usr/bin/env python3
"""R187 non-mutating reprobe for faithful external segmentation environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str], timeout: int = 60, env: dict[str, str] | None = None) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, env=env)
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"command": cmd, "error": repr(exc)}


def main() -> None:
    py = "/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python"
    conda = shutil.which("conda") or "/home/shenzeyu/miniconda3/bin/conda"
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    env = os.environ.copy()
    env["PATH"] = f"/usr/local/cuda/bin:/home/shenzeyu/miniconda3/bin:{env.get('PATH', '')}"
    env["CUDA_HOME"] = "/usr/local/cuda"
    probes = {
        "cwd": os.getcwd(),
        "paths": {
            "python": py,
            "conda": conda if Path(conda).exists() else None,
            "nvcc": nvcc if Path(nvcc).exists() else None,
            "cuda_home": "/usr/local/cuda",
        },
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], env=env),
        "python_torch": run([
            py,
            "-c",
            "import json, torch; print(json.dumps({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()}))",
        ], env=env),
        "nvcc_version": run([nvcc, "--version"], env=env) if Path(nvcc).exists() else {"missing": True},
        "conda_info": run([conda, "info", "--json"], env=env) if Path(conda).exists() else {"missing": True},
        "conda_env_list": run([conda, "env", "list"], env=env) if Path(conda).exists() else {"missing": True},
        "conda_dry_create_py310": run(
            [conda, "create", "--dry-run", "-y", "-n", "aris-maskdino-probe", "python=3.10"],
            timeout=120,
            env=env,
        ) if Path(conda).exists() else {"missing": True},
        "pip_detectron2_index": run([py, "-m", "pip", "index", "versions", "detectron2"], timeout=120, env=env),
        "pip_torch_index": run([py, "-m", "pip", "index", "versions", "torch"], timeout=120, env=env),
        "github_reachability": run([
            py,
            "-c",
            "import urllib.request,json; urls=['https://github.com/facebookresearch/Mask2Former','https://github.com/IDEA-Research/MaskDINO','https://github.com/facebookresearch/detectron2']; print(json.dumps({u: urllib.request.urlopen(u, timeout=10).status for u in urls}))",
        ], timeout=60, env=env),
        "disk": run(["df", "-h", ".", str(Path.home())], env=env),
    }
    can_create = (
        isinstance(probes["conda_dry_create_py310"], dict)
        and probes["conda_dry_create_py310"].get("returncode") == 0
    )
    nvcc_ok = isinstance(probes["nvcc_version"], dict) and probes["nvcc_version"].get("returncode") == 0
    probes["decision"] = {
        "isolated_conda_env_feasible": bool(can_create),
        "nvcc_visible_with_cuda_home": bool(nvcc_ok),
        "do_not_mutate_yolo_sam_gpu": True,
        "next_action": "create_isolated_maskdino_env_smoke" if can_create and nvcc_ok else "stay_blocked_or_choose_no_custom_cuda_arch",
        "risk_note": "System CUDA is 13.0 while current torch is cu118; faithful Detectron2/MaskDINO should be built in a fresh isolated env with a matching torch/CUDA stack.",
    }
    out = Path("outputs/analysis/r187_external_env_reprobe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probes, indent=2), encoding="utf-8")
    print(json.dumps(probes, indent=2))


if __name__ == "__main__":
    main()
