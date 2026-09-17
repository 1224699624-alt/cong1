#!/usr/bin/env python3
"""Synthetic checks for R257b frozen-center scale repair."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from train_r257_image_only_center_scale_detector import CenterScaleUNet
from train_r257b_scale_repair_comparison import basin_scales, method_gate, valid_region


def main()->None:
    info={"original_width":100.0,"original_height":50.0,"scale_x":1.0,"scale_y":1.0,"pad_x":0.0,"pad_y":25.0}; valid=valid_region(info,100); assert valid.sum()==5000 and not valid[0,0] and valid[25,0]
    mask=np.zeros((100,100),np.float32); mask[30:50,10:30]=1; mask[30:55,60:85]=1; proposals=[{"x":20.0,"y":40.0},{"x":72.0,"y":42.0}]; scales=basin_scales(mask,proposals,valid,0.5); assert np.allclose(scales,[20.0,25.0])
    passed=method_gate({"median_abs_log_scale_error":0.4,"close_endpoint_median_abs_log_scale_error":0.5,"contact_endpoint_median_abs_log_scale_error":0.55}); assert all(passed.values())
    failed=method_gate({"median_abs_log_scale_error":None,"close_endpoint_median_abs_log_scale_error":0.5,"contact_endpoint_median_abs_log_scale_error":0.55}); assert not all(failed.values())
    model=CenterScaleUNet(4).eval(); image=torch.rand(1,1,32,32)
    with torch.no_grad(): before=model(image)
    model.scale=torch.nn.Conv2d(4,1,1)
    with torch.no_grad(): after=model(image)
    assert torch.equal(before["center"],after["center"]) and torch.equal(before["mask"],after["mask"])
    print({"basin_scales":scales,"center_invariant":True}); print("R257b synthetic checks: PASS")


if __name__=="__main__": main()
