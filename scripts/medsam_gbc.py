#!/usr/bin/env python3
"""SCSegamba GBC adapted as a MedSAM image-embedding adapter.

Source design:
https://github.com/Karl1109/SCSegamba/blob/main/models/GBC.py
"""

from __future__ import annotations

import torch
from torch import nn


class BottConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size,
            stride,
            padding,
            groups=mid_channels,
            bias=False,
        )
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


def get_norm_layer(norm_type: str, channels: int, num_groups: int) -> nn.Module:
    if norm_type == "GN":
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    return nn.InstanceNorm3d(channels)


class SCSegambaGBC(nn.Module):
    """Official SCSegamba GBC structure for BxCxHxW feature maps."""

    def __init__(self, in_channels: int = 256, norm_type: str = "GN") -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU(),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)
        x = x1 * x2
        x = self.block4(x)
        return x + residual


class GBCAdapter(SCSegambaGBC):
    """Compatibility alias used by the MedSAM training and inference scripts."""

    def __init__(self, in_channels: int = 256, reduction: int = 8, norm_type: str = "GN") -> None:
        if reduction != 8:
            raise ValueError("The official SCSegamba GBC uses in_channels // 8 bottleneck channels.")
        super().__init__(in_channels=in_channels, norm_type=norm_type)
