import unittest
import torch
from r332_hard_source_supervision import hard_source_loss, local_negative_guard_loss


class HardSourceTests(unittest.TestCase):
    def test_negative_guard_support_and_gradient(self):
        z=torch.zeros((1,14,3,4),requires_grad=True);t=torch.zeros_like(z)
        teacher=torch.zeros_like(z,requires_grad=True);t[:,0]=1
        with torch.no_grad():teacher[:,2]=.25;teacher[:,3]=-.3
        valid=torch.ones((1,1,3,4),dtype=torch.bool);valid[:,:,:,0]=False
        loss,meta=local_negative_guard_loss(z,t,valid,teacher,100.)
        loss.backward()
        mask=(t<.5)&valid&(teacher.detach().abs()<.25)
        self.assertTrue(torch.equal(z.grad!=0,mask))
        self.assertTrue(torch.allclose(z.grad[mask],torch.full_like(z.grad[mask],.011)))
        self.assertIsNone(teacher.grad)
        self.assertEqual(meta['negative_members'],int(mask.sum()))

    def test_negative_guard_empty_support(self):
        z=torch.zeros((1,14,3,4),requires_grad=True)
        loss,meta=local_negative_guard_loss(z,torch.ones_like(z),torch.ones((1,1,3,4),dtype=torch.bool),z.detach(),100.)
        loss.backward();self.assertEqual(float(loss),0.);self.assertEqual(int(z.grad.count_nonzero()),0)

    def test_only_true_unselected_overlap_members_receive_gradient(self):
        z=torch.full((1,14,8,8),-8.,requires_grad=True);t=torch.zeros_like(z);t[:,0]=1;t[:,1]=1
        valid=torch.ones((1,1,8,8),dtype=torch.bool);valid[:,:,:,0]=False
        loss,meta=hard_source_loss(z,t,valid,torch.tensor([0,2,3,4]));loss.backward()
        self.assertGreater(meta['unselected_hard_members'],0)
        self.assertTrue(bool((z.grad[0,1,:,1:]<0).all()))
        self.assertEqual(int(torch.count_nonzero(z.grad[0,0])),0)
        self.assertEqual(int(torch.count_nonzero(z.grad[0,2:])),0)
        self.assertEqual(int(torch.count_nonzero(z.grad[:,:,:,0])),0)
        expected=meta['per_pixel_coefficient']*(z.detach().sigmoid()[0,1,:,1:]-1)
        self.assertTrue(torch.allclose(z.grad[0,1,:,1:],expected))

    def test_single_member_and_nonhard_targets_not_promoted(self):
        z=torch.full((1,14,8,8),-8.,requires_grad=True);t=torch.zeros_like(z);t[:,1]=1
        loss,meta=hard_source_loss(z,t,torch.ones((1,1,8,8),dtype=torch.bool),torch.tensor([0,2,3,4]))
        self.assertEqual(float(loss),0.);loss.backward();self.assertEqual(int(torch.count_nonzero(z.grad)),0)
        z2=torch.zeros_like(z,requires_grad=True);t[:,0]=1
        loss,meta=hard_source_loss(z2,t,torch.ones((1,1,8,8),dtype=torch.bool),torch.tensor([0,2,3,4]))
        self.assertEqual(meta['unselected_hard_members'],0)


if __name__=='__main__':unittest.main()
