"""Synthetic CPU checks for the O source-head response audit."""
from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_ram_o_source_head_response as audit  # noqa: E402


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


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = TinyBackbone()
        self.last_input: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value.detach().clone()
        stage4 = self.backbone.encoder(value)
        return self.backbone.decoder.seg_layers[4](stage4)


class TinyGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 1, 1, bias=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(image)


class RecordingLegacy:
    def __init__(self) -> None:
        self.call: tuple[torch.Tensor, torch.Tensor, object, int, torch.Tensor] | None = None

    def iterative_refine_o(
        self,
        image: torch.Tensor,
        source: torch.Tensor,
        refiner: object,
        steps: int,
        zero_o: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        self.call = (image.detach().clone(), source.detach().clone(), refiner, steps, zero_o.detach().clone())
        return [source, source, source], [torch.zeros_like(source), torch.zeros_like(source)]


class RamOSourceHeadResponseTests(unittest.TestCase):
    def _components(self) -> tuple[TinyModel, TinyGenerator, nn.Linear, torch.Tensor]:
        torch.manual_seed(19)
        model = TinyModel()
        generator = TinyGenerator()
        refiner = nn.Linear(2, 2)
        image = torch.linspace(-1.0, 1.0, 20).reshape(1, 1, 4, 5)
        with torch.no_grad():
            model.backbone.encoder.weight.fill_(0.05)
            model.backbone.decoder.seg_layers[4].weight.fill_(0.02)
            model.backbone.decoder.seg_layers[4].bias.fill_(0.1)
            generator.projection.weight.fill_(0.4)
            generator.projection.bias.fill_(0.3)
        return model, generator, refiner, image

    def test_strict_arm_load_and_runtime_arm_switch(self) -> None:
        model, generator, refiner, image = self._components()
        c0_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(c0_state, strict=True)
        attachment = audit.attach_after_c0(model, generator)
        with torch.no_grad():
            arm_weight = torch.full_like(attachment.adapter.weight, 0.25)
        payload = {
            "model": {name: value.detach().clone() for name, value in model.state_dict().items()},
            "refiner": {name: value.detach().clone() for name, value in refiner.state_dict().items()},
            "updates": 384,
        }
        payload["model"]["o_source_adapter.weight"] = arm_weight
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "arm.pth"
            torch.save(payload, path)
            audit.load_arm_pair_into(model, refiner, path, object())

        self.assertTrue(torch.equal(attachment.adapter.weight, arm_weight))
        attachment.use_real_o = False
        with torch.no_grad():
            zero_output = model(torch.cat((image, torch.zeros_like(image)), dim=1))
        self.assertFalse(attachment.last_trace["use_real_o"])
        self.assertEqual(attachment.last_trace["o"]["shape"], [1, 1, 4, 5])
        attachment.use_real_o = True
        with torch.no_grad():
            real_output = model(torch.cat((image, torch.zeros_like(image)), dim=1))
        self.assertTrue(attachment.last_trace["use_real_o"])
        self.assertGreater(float((real_output - zero_output).abs().sum()), 0.0)
        attachment.close()

    def test_forward_full14_has_no_gt_and_refiner_o_is_zero(self) -> None:
        model, generator, refiner, image = self._components()
        attachment = audit.attach_after_c0(model, generator)
        legacy = RecordingLegacy()
        source, states, deltas = audit.forward_full14(model, refiner, image, legacy)

        self.assertEqual(source.shape, (14, 4, 5))
        self.assertEqual(len(states), 3)
        self.assertEqual(len(deltas), 2)
        self.assertIsNotNone(legacy.call)
        call = legacy.call
        assert call is not None
        observed_image, _source, observed_refiner, steps, zero_o = call
        self.assertTrue(torch.equal(observed_image, image))
        self.assertIs(observed_refiner, refiner)
        self.assertEqual(steps, 2)
        self.assertTrue(torch.equal(zero_o, torch.zeros_like(image)))
        self.assertEqual(model.last_input.shape[1], 2)
        self.assertTrue(torch.equal(model.last_input[:, 1:], torch.zeros_like(model.last_input[:, 1:])))
        attachment.close()


if __name__ == "__main__":
    unittest.main()
