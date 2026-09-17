#!/usr/bin/env python3
"""R153 non-mutating feasibility probe for official Mask2Former/Mask DINO setup."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import paramiko


COMMANDS = {
    "whoami": "whoami",
    "pwd": "cd /home/shenzeyu/workspace/YOLO_SAM_generic_src && pwd",
    "python_version": "/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python -V",
    "torch_cuda": r"""cd /home/shenzeyu/workspace/YOLO_SAM_generic_src && /home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python - <<'PY'
import json, torch
print(json.dumps({
  "torch": torch.__version__,
  "cuda": torch.version.cuda,
  "cuda_available": torch.cuda.is_available(),
  "device_count": torch.cuda.device_count(),
  "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "capability_0": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
}, ensure_ascii=False))
PY""",
    "module_presence": r"""cd /home/shenzeyu/workspace/YOLO_SAM_generic_src && /home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python - <<'PY'
import importlib.util, json
mods = ["torch","torchvision","detectron2","fvcore","iopath","pycocotools","timm","transformers","mmcv","mmengine","mmdet","monai","nnunetv2","ninja"]
print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}, ensure_ascii=False, indent=2))
PY""",
    "nvidia_smi": "nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap --format=csv,noheader,nounits",
    "nvcc": "which nvcc && nvcc --version || true",
    "gcc": "which gcc && gcc --version | head -1 || true",
    "gxx": "which g++ && g++ --version | head -1 || true",
    "conda_info": "which conda || true; conda --version || true",
    "disk": "df -h /home/shenzeyu /home/shenzeyu/workspace 2>/dev/null || df -h /home/shenzeyu",
    "git": "git --version || true",
    "network_github": "python3 - <<'PY'\nimport urllib.request\nfor url in ['https://github.com/facebookresearch/Mask2Former', 'https://github.com/IDEA-Research/MaskDINO']:\n    try:\n        with urllib.request.urlopen(url, timeout=10) as r:\n            print(url, r.status)\n    except Exception as exc:\n        print(url, type(exc).__name__, str(exc)[:160])\nPY",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R153 remote feasibility probe.")
    parser.add_argument("--host", default="10.1.115.157")
    parser.add_argument("--user", default="shenzeyu")
    parser.add_argument("--password", default="40904090TWO")
    parser.add_argument("--output-json", default="outputs/analysis/r153_faithful_external_feasibility_probe.json")
    parser.add_argument("--output-md", default="research-workflow/refine-logs/R153_FAITHFUL_EXTERNAL_FEASIBILITY_PROBE.md")
    return parser.parse_args()


def run_remote(client: paramiko.SSHClient, command: str) -> dict[str, object]:
    stdin, stdout, stderr = client.exec_command(command, timeout=60)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return {"command": command, "stdout": out, "stderr": err, "exit_status": stdout.channel.recv_exit_status()}


def decision(results: dict[str, dict[str, object]]) -> dict[str, object]:
    module_text = str(results.get("module_presence", {}).get("stdout", "{}"))
    try:
        modules = json.loads(module_text)
    except json.JSONDecodeError:
        modules = {}
    nvcc_ok = bool(str(results.get("nvcc", {}).get("stdout", "")).strip())
    gcc_ok = bool(str(results.get("gcc", {}).get("stdout", "")).strip())
    github_text = str(results.get("network_github", {}).get("stdout", ""))
    github_ok = " 200" in github_text
    missing = [m for m in ["detectron2", "fvcore", "iopath", "pycocotools"] if not modules.get(m)]
    feasible = nvcc_ok and gcc_ok and github_ok
    return {
        "status": "go_isolated_install_probe" if feasible else "blocked_or_high_risk",
        "missing_core_modules": missing,
        "nvcc_available": nvcc_ok,
        "gcc_available": gcc_ok,
        "github_reachable": github_ok,
        "recommendation": (
            "Proceed to R154 isolated environment/dependency build probe; do not mutate yolo-sam-gpu."
            if feasible
            else "Do not install yet; resolve compiler/CUDA/network gaps or pivot architecture."
        ),
    }


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    dec = payload["decision"]
    lines = [
        "# R153 Faithful External Feasibility Probe",
        "",
        f"**Date**: {payload['timestamp']}",
        "",
        "## Decision",
        "",
        f"- Status: `{dec['status']}`",
        f"- Recommendation: {dec['recommendation']}",
        f"- Missing core modules: `{dec['missing_core_modules']}`",
        f"- NVCC available: `{dec['nvcc_available']}`",
        f"- GCC available: `{dec['gcc_available']}`",
        f"- GitHub reachable: `{dec['github_reachable']}`",
        "",
        "## Probe Outputs",
        "",
    ]
    for name, result in payload["results"].items():
        stdout = str(result.get("stdout", "")).strip()
        stderr = str(result.get("stderr", "")).strip()
        lines.extend([f"### {name}", "", "```text", stdout[:4000], "```", ""])
        if stderr:
            lines.extend(["stderr:", "", "```text", stderr[:2000], "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=args.password, timeout=20)
    try:
        results = {name: run_remote(client, cmd) for name, cmd in COMMANDS.items()}
    finally:
        client.close()
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": args.host,
        "policy": "non-mutating probe only; no package installation and no yolo-sam-gpu mutation",
        "results": results,
        "decision": decision(results),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(Path(args.output_md), payload)
    print(json.dumps(payload["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
