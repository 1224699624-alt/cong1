import unittest
import torch
from torch.nn import functional as F
from local_supervision_v3 import local_supervision


class LocalSupervisionTests(unittest.TestCase):
    def inputs(self):
        z = torch.zeros(1, 3, 4, 4, requires_grad=True)
        target = torch.zeros_like(z)
        target[:, :2, :2, :2] = 1
        known = torch.ones_like(z, dtype=torch.bool)
        union = target.any(1, keepdim=True)
        valid = torch.ones_like(union)
        seam = torch.arange(16.).reshape(1, 1, 4, 4) / 15
        return [z, target, known, union, valid, valid, seam, z.detach().sigmoid()]

    def test_same_image_has_both_states(self):
        args = self.inputs()
        terms, states = local_supervision(*args, enable_seam=True, enable_overlap=True)
        self.assertTrue(states['separation'].any())
        self.assertTrue(states['overlap'].any())
        self.assertFalse((states['separation'] & states['overlap']).any())
        (terms['separation'] + terms['overlap']).backward()
        self.assertTrue(torch.isfinite(args[0].grad).all())

    def test_unknown_is_not_negative(self):
        args = self.inputs(); args[2].zero_()
        terms, states = local_supervision(*args, enable_seam=True, enable_overlap=True)
        self.assertEqual(states['overlap_supervision'], 'unknown')
        terms['overlap'].backward()
        self.assertEqual(float(args[0].grad.abs().sum()), 0.)

    def test_partial_members_can_prove_positive_only(self):
        args = self.inputs(); args[2][:, 2] = False
        _, states = local_supervision(*args, enable_seam=True, enable_overlap=True)
        self.assertTrue(states['overlap_known'][0, 0, 0, 0])
        self.assertFalse(states['overlap_known'][0, 0, 3, 3])

    def test_padding_invariance_and_disabled_states(self):
        args = self.inputs()
        terms, states = local_supervision(*args, enable_seam=True, enable_overlap=True)
        padded = [F.pad(x, (0, 3, 0, 2)) for x in args]
        newer, ns = local_supervision(*padded, enable_seam=True, enable_overlap=True)
        for k in terms:
            self.assertTrue(torch.allclose(terms[k], newer[k]))
        self.assertEqual(states['masses'], ns['masses'])
        _, off = local_supervision(*args, enable_seam=False, enable_overlap=False)
        self.assertFalse(off['separation'].any() | off['overlap'].any())
        self.assertTrue(torch.equal(off['preserve'], args[5]))

    def test_empty_valid_is_finite_zero(self):
        args = self.inputs(); args[5].zero_()
        terms, _ = local_supervision(*args, enable_seam=True, enable_overlap=True)
        loss = sum(terms.values())
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward(); self.assertTrue(torch.isfinite(args[0].grad).all())


if __name__ == '__main__':
    unittest.main()
