"""Check that the proposed final H0 objective uses differentiable full14 refinement."""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r332_v31_port import R332RefinerWithO, iterative_refine_o


class FinalH0TargetTests(unittest.TestCase):
    def test_full14_two_step_parity_cap_and_gradient_route(self):
        torch.manual_seed(4421)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            image = torch.randn(1, 1, 32, 32)
            source = torch.randn(1, 14, 32, 32, requires_grad=True)
            refiner = R332RefinerWithO()
            with torch.no_grad():
                refiner.residual.bias.fill_(0.25)
            overlap = torch.zeros_like(image)
            h0 = torch.zeros_like(source, dtype=torch.bool)
            h0[0, 0, 4:12, 5:13] = True
            h0[0, 13, 16:20, 18:23] = True

            states, deltas = iterative_refine_o(image, source, refiner, 2, overlap)
            all_at_once, _ = iterative_refine_o(
                image, source, refiner, 2, overlap, indices=torch.arange(14)
            )
            self.assertEqual(len(states), 3)
            self.assertEqual(tuple(states[-1].shape), (1, 14, 32, 32))
            self.assertLessEqual(float((states[-1] - source).detach().abs().max()), 5.0 + 1e-5)
            self.assertTrue(torch.allclose(states[-1], all_at_once[-1], atol=1e-5, rtol=1e-5))
            self.assertGreater(float((states[-1] - source).detach().abs().max()), 0.01)

            final_h0 = F.softplus(-states[-1].float())[h0].mean()
            source_h0 = F.softplus(-source.float())[h0].mean()
            source_grad, refiner_grad = torch.autograd.grad(
                final_h0, (source, refiner.residual.weight), retain_graph=True
            )
            self.assertGreater(float(source_grad.abs().sum()), 0.0)
            self.assertGreater(float(refiner_grad.abs().sum()), 0.0)
            self.assertIsNone(torch.autograd.grad(
                source_h0, refiner.residual.weight, allow_unused=True
            )[0])
            self.assertEqual(len(deltas), 2)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
