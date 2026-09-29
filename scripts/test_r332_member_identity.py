import unittest
import torch
from r332_v31_port import R332RefinerWithO, iterative_refine_o
from r332_member_identity import IdentityContextRefiner,iterative_refine_identity,identity_permutation


class IdentityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(41)
        self.mature=R332RefinerWithO().eval()
        with torch.no_grad():self.mature.residual.weight.normal_(0,.03);self.mature.residual.bias.fill_(.1)
        self.image=torch.randn(1,1,32,32);self.source=torch.randn(1,14,32,32);self.o=torch.zeros_like(self.image)

    def test_zero_branch_preserves_mature_two_steps(self):
        before=torch.get_rng_state().clone();r=IdentityContextRefiner.from_mature(self.mature)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(torch.equal(r.residual.weight,self.mature.residual.weight))
        with torch.no_grad():
            old=iterative_refine_o(self.image,self.source,self.mature,2,self.o)[0]
            for mode in ['canonical','scrambled']:
                r.identity_mode=mode;new=iterative_refine_identity(self.image,self.source,r,2,self.o)[0]
                for a,b in zip(old,new):self.assertTrue(torch.equal(a,b))

    def test_new_branch_gradient_and_private_rng(self):
        r=IdentityContextRefiner.from_mature(self.mature);r.identity_mode='scrambled'
        before=torch.get_rng_state().clone();p=identity_permutation(self.image,'scrambled')
        self.assertTrue(torch.equal(p,identity_permutation(self.image,'scrambled')))
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        z=iterative_refine_identity(self.image,self.source,r,2,self.o,indices=torch.tensor([0,1,2,3]))[0][-1]
        z.square().mean().backward();self.assertTrue(torch.isfinite(r.relation_weight.grad).all());self.assertGreater(float(r.relation_weight.grad.abs().sum()),0)

    def test_relation_is_identity_sensitive_after_nonzero_weights(self):
        r=IdentityContextRefiner.from_mature(self.mature)
        with torch.no_grad():r.relation_weight.normal_(0,.1)
        with torch.no_grad():
            a=iterative_refine_identity(self.image,self.source,r,2,self.o)[0][-1]
            r.identity_mode='scrambled';b=iterative_refine_identity(self.image,self.source,r,2,self.o)[0][-1]
        self.assertGreater(float((a-b).abs().max()),0)

    def test_joint_relabeling_orientation(self):
        torch.manual_seed(8)
        weight=torch.randn(14,12,14);prob=self.source.sigmoid()[0]
        perm=torch.tensor([4,2,8,0,13,1,9,3,12,5,11,6,10,7]);query=torch.tensor([0,3,13])
        expected=torch.einsum('ncf,fhw->nchw',weight[query],prob)
        relabelled=weight[perm][:,:,perm]
        actual=torch.einsum('ncf,fhw->nchw',relabelled[torch.argsort(perm)[query]],prob[perm])
        self.assertTrue(torch.allclose(expected,actual,atol=2e-6,rtol=1e-5))


if __name__=='__main__':unittest.main()
