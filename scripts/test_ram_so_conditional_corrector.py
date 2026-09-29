"""Synthetic, CPU-only contract tests for the RAM S/O corrector."""
from __future__ import annotations

import inspect
import unittest

import torch

try:
    from ram_so_conditional_corrector import HEAD_INPUT_CHANNELS, PriorMaps, RAMSOConditionalCorrector
except ImportError:  # unittest discovery from the repository root
    from scripts.ram_so_conditional_corrector import HEAD_INPUT_CHANNELS, PriorMaps, RAMSOConditionalCorrector


class ConditionalCorrectorTests(unittest.TestCase):
    @staticmethod
    def _inputs(members: int = 4, height: int = 8, width: int = 9):
        torch.manual_seed(20260929)
        image = torch.randn(1, 1, height, width)
        z2 = torch.randn(1, members, height, width)
        hidden = torch.randn(members, 12, height, width)
        s = torch.rand(1, 1, height, width)
        o = torch.rand(1, 1, height, width)
        maps = PriorMaps(
            s=s,
            o=o,
            image_extent=(height, width),
            presence={"S": True, "O": True},
            metadata_hashes={"S": "s-hash", "O": "o-hash"},
        )
        return image, z2, hidden, s, o, maps

    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_contract_has_exact_21_channels_and_zero_identity_for_all_arms(self) -> None:
        image, z2, hidden, _, _, maps = self._inputs()
        corrector = RAMSOConditionalCorrector(bound=2.0).eval()
        self.assertEqual(corrector.head.input_projection.in_channels, HEAD_INPUT_CHANNELS)
        self.assertEqual(corrector.head.input_projection.out_channels, 32)
        self.assertEqual(corrector.head.final_layer.out_channels, 1)
        self.assertTrue(torch.equal(corrector.head.final_layer.weight, torch.zeros_like(corrector.head.final_layer.weight)))
        self.assertTrue(torch.equal(corrector.head.final_layer.bias, torch.zeros_like(corrector.head.final_layer.bias)))

        for arm in ("C00", "CS", "CO", "CSO"):
            final, delta = corrector(image, z2, hidden, maps, arm=arm, chunk_size=2)
            self.assertEqual(tuple(final.shape), tuple(z2.shape))
            self.assertEqual(tuple(delta.shape), tuple(z2.shape))
            self.assertTrue(torch.equal(delta, torch.zeros_like(delta)), arm)
            self.assertTrue(torch.equal(final, z2), arm)

    def test_signed_correction_is_member_specific_and_has_no_source_five_clamp(self) -> None:
        image, z2, hidden, _, _, maps = self._inputs(members=3, height=5, width=6)
        hidden.zero_()
        hidden[0, 0].fill_(1.0)
        hidden[1, 0].fill_(-1.0)
        hidden[2, 0].fill_(2.0)
        corrector = RAMSOConditionalCorrector(bound=2.0).eval()
        with torch.no_grad():
            corrector.head.input_projection.weight.zero_()
            corrector.head.input_projection.bias.zero_()
            # Channels 0..5 are the exact legacy feature block, channel 6 is
            # z2, and channel 7 is the first of the 12 mature dec1 channels.
            corrector.head.input_projection.weight[:, 7, 0, 0].fill_(1.0)
            corrector.head.final_layer.weight.fill_(0.1)

        _, positive = corrector(image, z2, hidden, maps, arm="CSO")
        self.assertGreater(float(positive[0, 0].mean().detach()), 0.0)
        self.assertTrue(torch.equal(positive[0, 1], torch.zeros_like(positive[0, 1])))
        self.assertGreater(float(positive[0, 2].mean().detach()), float(positive[0, 0].mean().detach()))
        self.assertLessEqual(float(positive.abs().max().detach()), 2.0)

        with torch.no_grad():
            corrector.head.final_layer.weight.fill_(-0.1)
        _, negative = corrector(image, z2, hidden, maps, arm="CSO")
        self.assertLess(float(negative[0, 0].mean().detach()), 0.0)
        self.assertLess(float(negative[0, 2].mean().detach()), 0.0)

        low_source = torch.full_like(z2, -10.0)
        final, _ = corrector(image, low_source, hidden, maps, arm="CSO")
        # The new bound is the only range restriction.  R332's old +/-5
        # iterative limit is not applied to the returned final logits.
        self.assertLess(float(final.min().detach()), -5.0)

    def test_reordering_logits_and_hidden_rows_reorders_outputs(self) -> None:
        image, z2, hidden, _, _, maps = self._inputs(members=4, height=6, width=7)
        corrector = RAMSOConditionalCorrector(bound=1.75).eval()
        with torch.no_grad():
            torch.manual_seed(91)
            corrector.head.input_projection.weight.normal_(0.0, 0.04)
            corrector.head.input_projection.bias.normal_(0.0, 0.02)
            corrector.head.final_layer.weight.normal_(0.0, 0.04)
            corrector.head.final_layer.bias.fill_(0.01)
        _, delta = corrector(image, z2, hidden, maps, arm="CSO", chunk_size=2)

        permutation = torch.tensor([2, 0, 3, 1])
        _, permuted_delta = corrector(
            image,
            z2[:, permutation],
            hidden[permutation],
            maps,
            arm="CSO",
            chunk_size=1,
        )
        self.assertTrue(torch.allclose(permuted_delta, delta[:, permutation], atol=1e-6, rtol=1e-6))

    def test_missing_required_maps_and_bad_ranges_fail(self) -> None:
        image, z2, hidden, s, o, maps = self._inputs()
        corrector = RAMSOConditionalCorrector(bound=2.0)
        c00_maps = PriorMaps(s=None, o=None, image_extent=(8, 9))
        final, delta = corrector(image, z2, hidden, c00_maps, arm="C00")
        self.assertTrue(torch.equal(final, z2))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

        with self.assertRaisesRegex(ValueError, "requires PriorMaps.s"):
            corrector(image, z2, hidden, PriorMaps(s=None, o=o, image_extent=(8, 9)), arm="CS")
        with self.assertRaisesRegex(ValueError, "requires PriorMaps.o"):
            corrector(image, z2, hidden, PriorMaps(s=s, o=None, image_extent=(8, 9)), arm="CO")
        with self.assertRaisesRegex(ValueError, "requires PriorMaps.o"):
            corrector(image, z2, hidden, PriorMaps(s=s, o=None, image_extent=(8, 9)), arm="CSO")

        bad_range = torch.full_like(s, 1.01)
        with self.assertRaisesRegex(ValueError, "soft map"):
            corrector(image, z2, hidden, PriorMaps(s=bad_range, o=o, image_extent=(8, 9)), arm="CS")
        with self.assertRaisesRegex(ValueError, "image_extent"):
            corrector(image, z2, hidden, PriorMaps(s=s, o=o, image_extent=(9, 9)), arm="CSO")
        final, delta = corrector(image, z2, hidden, PriorMaps(s=s, o=o, image_extent=(7, 8)), arm="CSO")
        self.assertTrue(torch.equal(final, z2))
        self.assertEqual(int(delta.count_nonzero()), 0)

    def test_no_gt_or_h0_input_and_frozen_inputs_are_untouched(self) -> None:
        image, z2, hidden, _, _, maps = self._inputs(members=2, height=5, width=5)
        signature = inspect.signature(RAMSOConditionalCorrector.forward)
        self.assertNotIn("target", signature.parameters)
        self.assertNotIn("gt", signature.parameters)
        self.assertNotIn("h0", signature.parameters)

        z2.requires_grad_()
        hidden.requires_grad_()
        image.requires_grad_()
        snapshots = (image.detach().clone(), z2.detach().clone(), hidden.detach().clone())
        corrector = RAMSOConditionalCorrector(bound=2.0)
        final, _ = corrector(image, z2, hidden, maps, arm="C00")
        self.assertTrue(torch.equal(image.detach(), snapshots[0]))
        self.assertTrue(torch.equal(z2.detach(), snapshots[1]))
        self.assertTrue(torch.equal(hidden.detach(), snapshots[2]))
        with self.assertRaises(TypeError):
            corrector(image, z2, hidden, maps, arm="C00", target=torch.zeros_like(z2))

    def test_first_backward_reaches_only_new_final_layer_and_optimizer_is_head_only(self) -> None:
        image, z2, hidden, _, _, maps = self._inputs(members=2, height=6, width=6)
        corrector = RAMSOConditionalCorrector(bound=2.0)
        with torch.no_grad():
            corrector.head.input_projection.weight.zero_()
            corrector.head.input_projection.bias.fill_(0.5)
        final, _ = corrector(image, z2, hidden, maps, arm="CSO")
        loss = final.square().mean()
        loss.backward()

        self.assertIsNotNone(corrector.head.final_layer.weight.grad)
        self.assertGreater(float(corrector.head.final_layer.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(corrector.head.final_layer.bias.grad)
        self.assertGreater(float(corrector.head.final_layer.bias.grad.abs().sum()), 0.0)
        self.assertTrue(torch.equal(
            corrector.head.input_projection.weight.grad,
            torch.zeros_like(corrector.head.input_projection.weight.grad),
        ))
        self.assertTrue(torch.equal(
            corrector.head.input_projection.bias.grad,
            torch.zeros_like(corrector.head.input_projection.bias.grad),
        ))

        optimizer_parameters = list(corrector.head_parameters())
        self.assertEqual({id(value) for value in optimizer_parameters}, {id(value) for value in corrector.head.parameters()})
        optimizer = torch.optim.SGD(optimizer_parameters, lr=0.1)
        self.assertEqual(
            {id(value) for group in optimizer.param_groups for value in group["params"]},
            {id(value) for value in corrector.head.parameters()},
        )
        # No optimizer.step(): this is a graph/parameter-routing test only.


if __name__ == "__main__":
    unittest.main()
