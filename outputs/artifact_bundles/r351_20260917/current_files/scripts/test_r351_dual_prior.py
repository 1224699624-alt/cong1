import unittest
import torch
from r351_dual_prior import SeamInputAdapter, OutputCompletionPrior, local_terms, gate_positive_completion


class DualPriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(351)

    def test_input_is_identity_but_trainable_and_prior_sensitive(self):
        model=SeamInputAdapter(); image=torch.randn(1,1,16,16)
        seam=torch.rand_like(image); overlap=torch.zeros_like(image)
        output=model(image,seam,overlap)
        self.assertTrue(torch.equal(output,image))
        output.square().mean().backward()
        self.assertGreater(float(model.net[-1].weight.grad.abs().sum()),0)
        with torch.no_grad():model.net[-1].bias.fill_(.3)
        self.assertFalse(torch.equal(model(image,seam,overlap),image))
        self.assertTrue(torch.equal(model(image,seam,torch.ones_like(image)),image))

    def test_unknown_membership_is_not_foreground_overlap(self):
        z=torch.zeros(1,1,8,8,requires_grad=True); target=torch.ones_like(z)
        terms,states=local_terms(z,target,torch.sigmoid(z.detach()),torch.rand_like(z),torch.ones_like(z),torch.ones_like(z),False)
        self.assertEqual(states['overlap_observed_mass'],0)
        self.assertEqual(float(terms['overlap']),0)
        terms['overlap'].backward()
        self.assertEqual(float(z.grad.abs().sum()),0)

    def test_true_multimembership_and_padding_mask(self):
        z=torch.zeros(1,2,8,8,requires_grad=True); target=torch.ones_like(z)
        valid=torch.ones(1,1,8,8); valid[:,:,:,4:]=0
        terms,states=local_terms(z,target,torch.sigmoid(z.detach()),torch.rand_like(valid),valid,valid,True)
        self.assertEqual(states['overlap_observed_mass'],1)
        terms['overlap'].backward()
        self.assertGreater(float(z.grad[:,:,:,:4].abs().sum()),0)
        self.assertEqual(float(z.grad[:,:,:,4:].abs().sum()),0)

    def test_refiner_zero_start_preserves_binary_and_multilabel(self):
        model=OutputCompletionPrior().eval(); image=torch.randn(1,1,16,16); seam=torch.rand_like(image)
        for channels in (1,3):
            z=torch.randn(1,channels,16,16)
            out=model.refine(image,z,seam,torch.rand_like(seam),steps=2)
            self.assertTrue(torch.equal(out,z))

    def test_distance_gain_cannot_hide_missing_prediction_failures(self):
        from train_r351_ram_dual_prior import native_gate
        baseline={'overall_dsc':.9,'overall_iou':.8,'overlap_msd_px':2.,'pair_msd_px':2.,
                  'overall_msd_fail_rate':0.,'overlap_msd_fail_rate':0.,'pair_msd_fail_rate':.001,
                  'overlap_dsc':.8,'overlap_nsd_2px':.7,'pair_nsd_2px':.7}
        candidate=dict(baseline,overlap_msd_px=1.8,pair_msd_px=1.8)
        self.assertTrue(native_gate(candidate,baseline)[0])
        candidate['pair_msd_fail_rate']=.002
        self.assertFalse(native_gate(candidate,baseline)[0])

    def test_positive_guard_uses_local_prior_not_targets(self):
        source=torch.zeros(1,1,2,2); refined=torch.ones_like(source)
        seam=torch.tensor([[[[0.,1.],[0.,1.]]]])
        out=gate_positive_completion(source,refined,seam,torch.zeros_like(source))
        self.assertTrue(torch.equal(out,1-seam))
        out=gate_positive_completion(source,refined,seam,torch.ones_like(source))
        self.assertTrue(torch.equal(out,refined))
        self.assertTrue(torch.equal(gate_positive_completion(source,-refined,seam,torch.zeros_like(source)),-refined))


if __name__=='__main__':unittest.main()
