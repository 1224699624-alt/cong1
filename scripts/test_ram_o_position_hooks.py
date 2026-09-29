"""CPU-only contract tests for :mod:`ram_o_position_hooks`."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ram_o_position_hooks import (  # noqa: E402
    ADAPTER_CHANNELS,
    ARM_INPUT_O,
    ARM_REFINER_O,
    ARM_ZERO,
    INPUT_FEATURE_CHANNELS,
    attach_position_hooks,
)


class _TinyConvWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, INPUT_FEATURE_CHANNELS, 3, padding=1, bias=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.conv(value)


class _TinyStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList([_TinyConvWrapper()])

    def forward(self, value: Tensor) -> Tensor:
        return self.convs[0](value)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stages = nn.ModuleList([nn.ModuleList([_TinyStage()])])

    def forward(self, value: Tensor) -> Tensor:
        return self.stages[0][0](value)


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TinyEncoder()

    def forward(self, value: Tensor) -> Tensor:
        return self.encoder(value)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()

    def forward(self, value: Tensor) -> Tensor:
        return self.backbone(value)


class _TinyRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Module()
        self.enc1.net = nn.Sequential(
            nn.Conv2d(7, ADAPTER_CHANNELS, 3, padding=1, bias=True),
            nn.SiLU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.enc1.net(value)


def _components() -> tuple[_TinyModel, _TinyRefiner, Tensor, Tensor]:
    torch.manual_seed(17)
    model = _TinyModel()
    refiner = _TinyRefiner()
    image = torch.randn(1, 1, 8, 9)
    model_input = torch.cat((image, torch.zeros_like(image)), dim=1)
    feature = torch.randn(3, 7, 8, 9)
    feature[:, 6:7].zero_()
    return model, refiner, model_input, feature


def _input_conv(model: _TinyModel) -> nn.Conv2d:
    return model.backbone.encoder.stages[0][0].convs[0].conv


class RamOPositionHookTests(unittest.TestCase):
    def test_zero_initialization_and_mature_weight_parity(self) -> None:
        model, refiner, model_input, feature = _components()
        input_weight = _input_conv(model).weight.detach().clone()
        refiner_weight = refiner.enc1.net[0].weight.detach().clone()
        base_model = model(model_input)
        base_refiner = refiner(feature)

        attachment = attach_position_hooks(model, refiner, ARM_ZERO)
        self.assertEqual(tuple(attachment.input_adapter.weight.shape), (12, 1, 3, 3))
        self.assertEqual(tuple(attachment.refiner_adapter.weight.shape), (12, 1, 3, 3))
        self.assertTrue(torch.equal(attachment.input_adapter.weight, torch.zeros_like(attachment.input_adapter.weight)))
        self.assertTrue(torch.equal(attachment.refiner_adapter.weight, torch.zeros_like(attachment.refiner_adapter.weight)))
        projection = attachment.early_projection.weight[:, :, 0, 0]
        self.assertTrue(torch.allclose(projection.transpose(0, 1) @ projection, torch.eye(12), atol=1e-6, rtol=0.0))

        attachment.set_o(torch.randn(1, 1, 8, 9))
        hooked_model = model(model_input)
        hooked_refiner = refiner(feature)
        self.assertTrue(torch.equal(base_model, hooked_model))
        self.assertTrue(torch.equal(base_refiner, hooked_refiner))
        self.assertTrue(torch.equal(input_weight, _input_conv(model).weight))
        self.assertTrue(torch.equal(refiner_weight, refiner.enc1.net[0].weight))
        self.assertEqual(attachment.stats["input_hook_calls"], 1)
        self.assertEqual(attachment.stats["refiner_hook_calls"], 1)
        self.assertEqual(attachment.stats["input_applied_calls"], 0)
        self.assertEqual(attachment.stats["refiner_applied_calls"], 0)
        attachment.close()

    def test_input_arm_uses_o_only_at_early_position_and_has_gradients(self) -> None:
        model, refiner, model_input, feature = _components()
        attachment = attach_position_hooks(model, refiner, ARM_INPUT_O)
        o = torch.rand(1, 1, 8, 9, requires_grad=True)
        attachment.set_o(o)
        output = model(model_input)
        refined = refiner(feature)
        loss = output.square().mean() + refined.square().mean()
        loss.backward()

        self.assertGreater(float(attachment.input_adapter.weight.grad.abs().sum()), 0.0)
        self.assertTrue(torch.equal(attachment.refiner_adapter.weight.grad, torch.zeros_like(attachment.refiner_adapter.weight.grad)))
        self.assertIsNone(o.grad)
        self.assertEqual(attachment.stats["input_applied_calls"], 1)
        self.assertEqual(attachment.stats["refiner_applied_calls"], 0)
        attachment.close()

    def test_refiner_arm_repeats_o_for_member_chunks_and_has_gradients(self) -> None:
        model, refiner, model_input, feature = _components()
        attachment = attach_position_hooks(model, refiner, ARM_REFINER_O)
        o = torch.rand(1, 1, 8, 9, requires_grad=True)
        attachment.set_o(o)
        model_output = model(model_input)
        # R332's chunked refiner has one O per image and several member rows.
        refined = refiner(feature)
        self.assertEqual(tuple(refined.shape), (3, 12, 8, 9))
        loss = model_output.square().mean() + refined.square().mean()
        loss.backward()

        self.assertTrue(torch.equal(attachment.input_adapter.weight.grad, torch.zeros_like(attachment.input_adapter.weight.grad)))
        self.assertGreater(float(attachment.refiner_adapter.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(o.grad)
        self.assertEqual(attachment.stats["input_applied_calls"], 0)
        self.assertEqual(attachment.stats["refiner_applied_calls"], 1)
        attachment.close()

    def test_zero_arm_adapters_receive_explicit_zero_gradients(self) -> None:
        model, refiner, model_input, feature = _components()
        attachment = attach_position_hooks(model, refiner, ARM_ZERO)
        attachment.set_o(torch.rand(1, 1, 8, 9))
        loss = model(model_input).square().mean() + refiner(feature).square().mean()
        loss.backward()
        self.assertIsNotNone(attachment.input_adapter.weight.grad)
        self.assertIsNotNone(attachment.refiner_adapter.weight.grad)
        self.assertTrue(torch.equal(attachment.input_adapter.weight.grad, torch.zeros_like(attachment.input_adapter.weight.grad)))
        self.assertTrue(torch.equal(attachment.refiner_adapter.weight.grad, torch.zeros_like(attachment.refiner_adapter.weight.grad)))
        attachment.close()

    def test_close_removes_both_hooks_and_restores_mature_forward(self) -> None:
        model, refiner, model_input, feature = _components()
        base_model = model(model_input)
        base_refiner = refiner(feature)
        attachment = attach_position_hooks(model, refiner, ARM_INPUT_O)
        attachment.set_o(torch.ones(1, 1, 8, 9))
        with torch.no_grad():
            attachment.input_adapter.weight.fill_(0.25)
            attachment.refiner_adapter.weight.fill_(0.25)
        self.assertFalse(torch.equal(base_model, model(model_input)))
        attachment.close()
        self.assertTrue(torch.equal(base_model, model(model_input)))
        self.assertTrue(torch.equal(base_refiner, refiner(feature)))
        self.assertTrue(attachment.stats["closed"])
        self.assertEqual(len(_input_conv(model)._forward_hooks), 0)
        self.assertEqual(len(refiner.enc1.net[0]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
