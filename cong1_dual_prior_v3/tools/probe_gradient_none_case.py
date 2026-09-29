#!/usr/bin/env python3
"""Read-only CPU regression for the R349 partially unused-gradient case.

Exit 0: the projected auxiliary no longer conflicts with the base direction.
Exit 1: the regression is present. Exit 2: setup/import/execution failed.
This is not a segmentation experiment and does not read medical data/weights.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

AUDITED_GIT_BLOB = "7da1eba1b4f39ca8b07869ba0eeb70fc1731022a"


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", type=Path, help="Repository root; load scripts/r349_parameter_gradient_gate.py")
    source.add_argument("--source", type=Path, help="Explicit source file, e.g. the supplied read-only snapshot")
    parser.add_argument("--output", type=Path, help="Optional NEW JSON output file; existing files are not overwritten")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    try:
        if not math.isfinite(args.tolerance) or args.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        path = (args.source if args.source is not None else args.repo / "scripts/r349_parameter_gradient_gate.py").resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite: {args.output}")
        import torch
        name = "_audited_r349_gradient_module"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        # dataclasses resolve their declaring module through sys.modules.
        sys.modules[name] = module
        spec.loader.exec_module(module)
        x = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        y = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        base_loss = x + y
        auxiliary_loss = -x  # y is unused by this auxiliary; its gradient is None.
        combined, stats = module.route_parameter_gradients(
            [x, y], base_loss, [("partial_aux", auxiliary_loss, 1.0)], max_aux_ratio=10.0
        )
        if len(combined) != 2 or any(g is None for g in combined):
            raise RuntimeError("Expected two combined gradients; both parameters are used by base_loss")
        combined_values = [float(g.detach()) for g in combined]
        applied_aux = [value - 1.0 for value in combined_values]
        dot = sum(applied_aux)  # dot with g_base=(1,1)
        finite = all(math.isfinite(value) for value in combined_values + applied_aux)
        passed = finite and dot >= -args.tolerance
        report = {
            "kind": "CPU_SYNTHETIC_REGRESSION_NOT_MODEL_EXPERIMENT",
            "source_path": str(path),
            "source_git_blob_sha": git_blob_sha(path),
            "audited_git_blob_sha": AUDITED_GIT_BLOB,
            "matches_audited_snapshot": git_blob_sha(path) == AUDITED_GIT_BLOB,
            "torch": torch.__version__,
            "base_gradient": [1.0, 1.0],
            "aux_gradient": [-1.0, None],
            "correct_full_space_projected_aux": [-0.5, 0.5],
            "combined_gradient": combined_values,
            "applied_auxiliary": applied_aux,
            "applied_aux_dot_base": dot,
            "stats": [dataclasses.asdict(item) if dataclasses.is_dataclass(item) else vars(item) for item in stats],
            "tolerance": args.tolerance,
            "passed": passed,
            "status": "PASS" if passed else "FAIL_PARTIAL_NONE_PROJECTION",
            "limitation": "A passing synthetic test does not establish model non-degradation or full routing correctness.",
        }
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text + "\n")
        return 0 if passed else 1
    except Exception as exc:
        print(json.dumps({"status": "PROBE_ERROR", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
