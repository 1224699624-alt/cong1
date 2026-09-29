import hashlib,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from structural_prior_v3 import build_backbone_input,StructuralPriorV3,ImageOnlyOverlapPrior,OutputOverlapAdapter


class IsolationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)
        self.x=torch.randn(1,1,16,16);self.s=torch.rand_like(self.x)

    def test_joint_input_not_modified_image(self):
        for arm in 'ABCD':
            x=build_backbone_input(self.x,self.s,arm)
            self.assertTrue(torch.equal(x[:,:1],self.x))
            self.assertTrue(torch.equal(x[:,1:2],self.s if arm in 'BD' else torch.zeros_like(self.s)))

    def test_A_B_no_overlap_call_or_file_read(self):
        forbidden=nn.Module()
        for arm in 'AB':
            model=StructuralPriorV3(arm,overlap_generator=forbidden,output_refiner=forbidden)
            with patch('builtins.open',side_effect=AssertionError('file read')):
                z=torch.randn(1,2,16,16);self.assertIs(model.refine_output(self.x,z),z)
                model.backbone_input(self.x,self.s)
            self.assertIsNone(model.overlap_generator)

    def test_C_seam_perturbation_and_file_independence(self):
        gen=ImageOnlyOverlapPrior().eval();ref=OutputOverlapAdapter().eval()
        with torch.no_grad():ref.residual.weight.fill_(.01)
        p=StructuralPriorV3('C',overlap_generator=gen,output_refiner=ref,mechanical_test_only=True)
        backbone=nn.Conv2d(2,2,1)
        with patch('builtins.open',side_effect=AssertionError('file read')):
            z1=backbone(p.backbone_input(self.x,self.s));z2=backbone(p.backbone_input(self.x,object()))
            y1=p.refine_output(self.x,z1);y2=p.refine_output(self.x,z2)
        self.assertTrue(torch.equal(y1,y2))

    def test_O_independent_channel_and_gradient(self):
        ref=OutputOverlapAdapter();z=torch.randn(1,2,16,16);o=torch.rand_like(self.x)
        y=ref.refine(self.x,z,o);self.assertTrue(torch.equal(y,z))
        y.square().mean().backward();self.assertGreater(float(ref.residual.weight.grad.abs().sum()),0.)
        with torch.no_grad():ref.residual.weight.fill_(.02)
        a=ref.refine(self.x,z,torch.zeros_like(o));b=ref.refine(self.x,z,torch.ones_like(o))
        self.assertGreater(float((a-b).detach().abs().max()),0.)

    def test_D_reachable_both_paths(self):
        gen=ImageOnlyOverlapPrior();ref=OutputOverlapAdapter();p=StructuralPriorV3('D',overlap_generator=gen,output_refiner=ref,mechanical_test_only=True)
        backbone=nn.Conv2d(2,2,1);y=p.refine_output(self.x,backbone(p.backbone_input(self.x,self.s)))
        y.square().mean().backward()
        self.assertGreater(float(backbone.weight.grad[:,1].abs().sum()),0.)
        self.assertGreater(float(ref.residual.weight.grad.abs().sum()),0.)

    def test_unvalidated_weights_blocked(self):
        for arm in 'CD':
            with self.assertRaises(RuntimeError):StructuralPriorV3(arm)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):build_backbone_input(self.x,self.s[:,:,:8],'B')

    def test_legacy_conversion_is_initialization_not_trained_prior(self):
        from r351_dual_prior import ImageOverlapPrior
        old=ImageOverlapPrior();new=ImageOnlyOverlapPrior()
        report=new.initialize_from_legacy(old.state_dict())
        self.assertEqual(report['status'],'BLOCKED_NOT_ADAPTED')
        self.assertTrue(torch.equal(new.first.net[0].weight,old.first.net[0].weight[:,:1]))
        self.assertEqual(new(self.x).shape,self.x.shape)

    def test_inference_signatures_have_no_ground_truth(self):
        import inspect
        for method in [build_backbone_input,ImageOnlyOverlapPrior.forward,OutputOverlapAdapter.refine,StructuralPriorV3.refine_output]:
            names=set(inspect.signature(method).parameters)
            self.assertFalse(names & {'target','mask','labels','gt'})


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IsolationTests))
    root=Path(__file__).resolve().parents[1];out=root/'outputs/v3_round2';out.mkdir(parents=True,exist_ok=True)
    (out/'ARM_INFORMATION_ISOLATION.json').write_text(json.dumps({'status':'PASS' if result.wasSuccessful() else 'FAIL',
        'scope':'CPU mechanical tests with random test modules; not trained prior validation','tests_run':result.testsRun,
        'real_image_only_overlap_weights':'BLOCKED_NOT_VERIFIED','command':f'{sys.executable} scripts/test_structural_prior_v3.py',
        'version_sha256':hashlib.sha256((root/'scripts/structural_prior_v3.py').read_bytes()).hexdigest()},indent=2))
    sys.exit(not result.wasSuccessful())
