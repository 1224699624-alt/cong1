import inspect
import unittest

import torch

from final_prior_plugin import (
    BackboneAdapter,
    StructuralPriorPlugin,
    variant_config,
    zero_native_prior_channel,
)


class FinalPluginTests(unittest.TestCase):
    def test_variant_matrix(self):
        self.assertEqual(variant_config("A")["name"], "native_continue")
        self.assertTrue(variant_config("D")["enable_input_prior"])
        self.assertTrue(variant_config("D")["enable_output_prior"])

    def test_core_aliases_are_dataset_native(self):
        self.assertEqual(variant_config("T0")["name"], "native_continue")
        self.assertEqual(variant_config("T1")["name"], "seam_only")
        self.assertEqual(variant_config("R0")["name"], "native_continue")
        self.assertEqual(variant_config("R1")["name"], "overlap_only")

    def test_zero_native_channel(self):
        image = torch.randn(2, 1, 8, 9)
        seam = torch.rand_like(image)
        native = zero_native_prior_channel(image, seam)
        self.assertTrue(torch.equal(native[:, 0], image[:, 0]))
        self.assertEqual(float(native[:, 1].abs().max()), 0.0)

    def test_forward_contract_has_no_target(self):
        self.assertNotIn("target", inspect.signature(StructuralPriorPlugin.forward).parameters)

    def test_output_identity_keeps_gradient_path(self):
        torch.manual_seed(1)
        plugin = StructuralPriorPlugin(enable_input=False, enable_output=True)
        image = torch.randn(1, 1, 12, 12)
        logits = torch.randn(1, 2, 12, 12)
        seam = torch.rand_like(image)
        out = plugin.refine_output(image, logits, seam, torch.zeros_like(image))
        self.assertTrue(torch.equal(out, logits))
        out.square().mean().backward()
        self.assertIsNotNone(plugin.output_refiner.residual.weight.grad)
        self.assertGreater(float(plugin.output_refiner.residual.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
