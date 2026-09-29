"""CPU checks for the RAM H0 update-direction audit."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_ram_h0_update_direction import (  # noqa: E402
    assert_loss_parity,
    build_masks,
    component_losses,
    weighted_loss,
)


class RamH0UpdateDirectionTests(unittest.TestCase):
    def test_reconstructed_full14_loss_matches_runner_on_padded_crop(self) -> None:
        torch.manual_seed(4421)
        final = torch.randn(1, 14, 8, 9, requires_grad=True)
        target = (torch.rand(1, 14, 8, 9) > 0.72).float()
        valid = torch.ones(1, 1, 8, 9)
        valid[:, :, 6:, :] = 0
        valid[:, :, :, 7:] = 0
        target = target * valid

        differences = assert_loss_parity(final, target, valid, (6, 7))
        self.assertLessEqual(max(differences.values()), 2e-6)
        components = component_losses(final, target, valid, (6, 7))
        total = weighted_loss(components)
        total.backward()
        self.assertTrue(torch.isfinite(final.grad).all())
        self.assertGreater(float(final.grad.abs().sum()), 0.0)

    def test_h0_negative_and_c0_correct_positive_masks_are_fixed(self) -> None:
        target = torch.zeros(1, 14, 5, 6)
        target[0, 0, 1, 1] = 1
        target[0, 1, 1, 1] = 1
        target[0, 2, 2, 2] = 1
        valid = torch.ones(1, 1, 5, 6)
        valid[:, :, 4:, 5:] = 0
        c0_final = torch.full_like(target, -2.0)
        c0_final[0, 2, 2, 2] = 1.0  # a C0-correct positive
        c0_final[0, 3, 1, 1] = -1.0  # co-located true negative at H0 union
        c0_final[0, 4, 0, 0] = 1.0  # ordinary correct positive only if target is set
        h0 = torch.zeros_like(target, dtype=torch.bool)
        h0[0, 0, 1, 1] = True
        h0[0, 1, 1, 1] = True

        masks = build_masks(target, valid, c0_final, h0)
        self.assertEqual(int(masks["H"].sum()), 2)
        self.assertEqual(int(masks["H"].any(dim=1).sum()), 1)
        self.assertEqual(int(masks["N"].sum()), 12)  # 12 other true-negative members at the H0 pixel
        self.assertEqual(int(masks["C"].sum()), 1)
        self.assertFalse(bool((masks["H"] & ~target.bool()).any()))
        self.assertFalse(bool((masks["H"] & ~(valid > 0.5)).any()))
        self.assertFalse(bool((masks["C"] & ~(target.bool())).any()))


if __name__ == "__main__":
    unittest.main()
