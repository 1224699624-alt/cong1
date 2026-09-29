import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_ram_so_o_pair import low_targets,bce_loss,thresholds_calibrate


class OPairTests(unittest.TestCase):
    def test_valid_area_preserves_soft_target_and_dedup(self):
        y=torch.zeros(1,14,4,4)
        y[:,0,:2,:2]=1;y[:,1,:2,:2]=1;y[:,2,0,0]=1
        valid=torch.ones(1,1,4,4,dtype=torch.bool);valid[:,:,-1,:]=False
        h=torch.zeros(1,1,4,4,dtype=torch.bool);h[:,:,0,0]=True
        t,v,hg=low_targets(y,valid,h,size=2)
        self.assertTrue(bool((hg<=t).all()))
        self.assertEqual(float(hg[0,0,0,0]),.25)
        self.assertEqual(float(t[0,0,0,0]),1.)
        self.assertEqual(float(v[0,0,1,0]),.5)

    def test_beta0_equal_reference_and_h_not_target(self):
        z=torch.tensor([[[[.1,-.9],[1.,-2.]]]],requires_grad=True)
        t=torch.tensor([[[[.8,0.],[1.,0.]]]])
        h=torch.tensor([[[[.2,0.],[.7,0.]]]])
        v=torch.ones_like(t)
        ref=(F.binary_cross_entropy_with_logits(z,t,reduction='none')*v).sum()/v.sum()
        self.assertTrue(torch.equal(ref,bce_loss(z,t,v,h,0)))
        expected=(v*(1+h)*F.binary_cross_entropy_with_logits(z,t,reduction='none')).sum()/(v*(1+h)).sum()
        self.assertTrue(torch.equal(expected,bce_loss(z,t,v,h,1)))
        self.assertFalse(torch.equal(bce_loss(z,t,v,h,1),bce_loss(z,h,v,h,1)))

    def test_calibration_ties_do_not_exceed_budget(self):
        scores=np.asarray([.9,.9,.8,.7,.7,.1],dtype=np.float32)
        threshold=thresholds_calibrate(scores,3)
        self.assertEqual(int((scores>=threshold).sum()),3)
        threshold=thresholds_calibrate(scores,1)
        self.assertEqual(int((scores>=threshold).sum()),0)


if __name__=='__main__':unittest.main()
