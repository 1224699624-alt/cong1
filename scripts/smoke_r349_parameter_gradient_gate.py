"""Deterministic unit smoke for R349 parameter-space gradient routing."""
from __future__ import annotations

import json
import torch

from r349_parameter_gradient_gate import assign_gradients, route_parameter_gradients


def main() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    base = parameter[0] + parameter[1]
    opposing = -2 * parameter[0] - parameter[1]
    helpful = parameter[1]
    gradients, stats = route_parameter_gradients(
        [parameter], base, [("opposing", opposing, 1.0), ("helpful", helpful, 1.0)], max_aux_ratio=.15)
    assign_gradients([parameter], gradients)
    for row in stats:
        assert row.cosine_after >= -1e-6
        assert row.applied_norm <= .15 * (2 ** .5) + 1e-6
    assert stats[0].conflicted and not stats[1].conflicted
    assert torch.isfinite(parameter.grad).all()
    print(json.dumps({"passed": True, "combined_gradient": parameter.grad.tolist(),
                      "stats": [vars(row) for row in stats]}, indent=2))


if __name__ == "__main__":
    main()
