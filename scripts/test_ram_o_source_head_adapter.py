"""CPU tests for ``ram_o_source_head_adapter``.

These tests use a tiny native-resolution decoder shaped like the mature RAM
model.  They exercise the hook and state contract without loading a checkpoint
or allocating the full GPU model.
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ram_o_source_head_adapter import attach_o_source_head  # noqa: E402


class TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 1, 1, bias=True)
        self.register_buffer("sentinel", torch.tensor([17.0]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(image) + self.sentinel.view(1, 1, 1, 1) * 0.0


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seg_layers = nn.ModuleList([nn.Identity() for _ in range(5)])
        self.seg_layers[4] = nn.Conv2d(32, 14, 1, bias=True)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(2, 32, 1, bias=False)
        self.decoder = TinyDecoder()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        stage4 = self.encoder(value)
        return self.decoder.seg_layers[4](stage4)


class TinyRamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = TinyBackbone()
        self.register_buffer("sentinel", torch.tensor([23.0]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.backbone(value) + self.sentinel.view(1, 1, 1, 1) * 0.0


def _make_components() -> tuple[TinyRamModel, TinyGenerator, torch.Tensor]:
    torch.manual_seed(7)
    model = TinyRamModel()
    generator = TinyGenerator()
    image = torch.linspace(-1.0, 1.0, 4 * 5).reshape(1, 1, 4, 5)
    value = torch.cat((image, torch.zeros_like(image)), dim=1)
    with torch.no_grad():
        model.backbone.encoder.weight.fill_(0.05)
        model.backbone.decoder.seg_layers[4].weight.fill_(0.02)
        model.backbone.decoder.seg_layers[4].bias.fill_(0.1)
        generator.projection.weight.fill_(0.4)
        generator.projection.bias.fill_(0.3)
    return model, generator, value


class RamOSourceHeadAdapterTests(unittest.TestCase):
    def test_zero_initialized_real_and_zero_arms_are_exactly_parity(self) -> None:
        base, generator, value = _make_components()
        # Strict C0 loading happens before attachment.  The pre-attachment
        # state keys/parameter names are the contract inherited by the runner.
        c0_state = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}
        original_names = [name for name, _ in base.named_parameters()]
        restored, restored_generator, _ = _make_components()
        restored.load_state_dict(c0_state, strict=True)
        self.assertEqual(original_names, [name for name, _ in restored.named_parameters()])

        zero = attach_o_source_head(base, generator, use_real_o=False)
        real = attach_o_source_head(restored, restored_generator, use_real_o=True)
        self.assertTrue(torch.equal(zero.adapter.weight, torch.zeros_like(zero.adapter.weight)))
        self.assertTrue(torch.equal(real.adapter.weight, torch.zeros_like(real.adapter.weight)))
        names_after = [name for name, _ in restored.named_parameters()]
        self.assertEqual(names_after[:len(original_names)], original_names)
        self.assertEqual(names_after[-1], "o_source_adapter.weight")
        self.assertEqual(tuple(real.adapter.weight.shape), (14, 32, 1, 1))

        source_zero = base(value)
        source_real = restored(value)
        self.assertTrue(torch.equal(source_zero, source_real))
        self.assertTrue(torch.equal(zero.last_trace["o"]["mean"], 0.0) if isinstance(zero.last_trace["o"]["mean"], torch.Tensor) else zero.last_trace["o"]["mean"] == 0.0)
        self.assertTrue(real.last_trace["use_real_o"])
        self.assertEqual(real.last_trace["o"]["shape"], [1, 1, 4, 5])

    def test_adapter_change_is_member_specific_and_zero_arm_is_inert(self) -> None:
        real_model, real_generator, value = _make_components()
        zero_model, zero_generator, _ = _make_components()
        real = attach_o_source_head(real_model, real_generator, use_real_o=True)
        zero = attach_o_source_head(zero_model, zero_generator, use_real_o=False)
        with torch.no_grad():
            before_real = real_model(value)
            before_zero = zero_model(value)
            real.adapter.weight[3, 7, 0, 0] = 0.75
            zero.adapter.weight[3, 7, 0, 0] = 0.75
            after_real = real_model(value)
            after_zero = zero_model(value)
        delta_real = after_real - before_real
        delta_zero = after_zero - before_zero
        self.assertGreater(float(delta_real[:, 3].abs().sum()), 0.0)
        self.assertTrue(torch.equal(delta_real[:, :3], torch.zeros_like(delta_real[:, :3])))
        self.assertTrue(torch.equal(delta_real[:, 4:], torch.zeros_like(delta_real[:, 4:])))
        self.assertTrue(torch.equal(delta_zero, torch.zeros_like(delta_zero)))
        self.assertGreater(real.last_trace["adapter_response"]["abs_mean"], 0.0)
        self.assertEqual(zero.last_trace["adapter_response"]["abs_mean"], 0.0)

    def test_real_o_gradients_reach_adapter_and_source_but_not_generator(self) -> None:
        model, generator, value = _make_components()
        attachment = attach_o_source_head(model, generator, use_real_o=True)
        output = model(value)
        loss = output.square().mean()
        loss.backward()
        self.assertIsNotNone(attachment.adapter.weight.grad)
        self.assertGreater(float(attachment.adapter.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.backbone.encoder.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.backbone.decoder.seg_layers[4].weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in generator.parameters()))

    def test_zero_o_arm_has_zero_adapter_gradient_but_source_still_trains(self) -> None:
        model, generator, value = _make_components()
        attachment = attach_o_source_head(model, generator, use_real_o=False)
        output = model(value)
        output.square().mean().backward()
        self.assertIsNotNone(attachment.adapter.weight.grad)
        self.assertTrue(torch.equal(attachment.adapter.weight.grad, torch.zeros_like(attachment.adapter.weight.grad)))
        self.assertGreater(float(model.backbone.encoder.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.backbone.decoder.seg_layers[4].weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in generator.parameters()))

    def test_forward_does_not_change_rng_or_buffers_and_trace_is_detached(self) -> None:
        model, generator, value = _make_components()
        attachment = attach_o_source_head(model, generator, use_real_o=True)
        model_buffers = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}
        generator_buffers = {name: tensor.detach().clone() for name, tensor in generator.named_buffers()}
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        model(value)
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        for name, before in model_buffers.items():
            self.assertTrue(torch.equal(before, dict(model.named_buffers())[name]))
        for name, before in generator_buffers.items():
            self.assertTrue(torch.equal(before, dict(generator.named_buffers())[name]))
        trace = attachment.last_trace
        self.assertIsNotNone(trace)
        self.assertTrue(all(isinstance(value, (int, float, str, bool, list, type(None)))
                            for block in trace.values() if isinstance(block, dict)
                            for value in block.values()))
        attachment.close()
        self.assertEqual(len(model.backbone.decoder.seg_layers[4]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
