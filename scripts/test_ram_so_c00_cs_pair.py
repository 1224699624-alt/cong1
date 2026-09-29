"""Focused full14 loss and frozen-head experiment invariants."""
import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_ram_so_c00_cs_pair import complete_loss

class C00CSTests(unittest.TestCase):
    def test_native_loss_ignores_padded_border_and_reaches_positive_negative_members(self):
        torch.set_num_threads(1)
        logits=torch.zeros(1,14,16,16,requires_grad=True)
        target=torch.zeros_like(logits);target.data[:,0,2:8,2:8]=1;target.data[:,1,4:10,4:10]=1
        valid=torch.zeros(1,1,16,16,dtype=torch.bool);valid[:,:,:12,:12]=True
        loss,_=complete_loss(logits,target,valid,(12,12));loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad[:,0,:12,:12].abs().sum()),0)
        self.assertGreater(float(logits.grad[:,13,:12,:12].abs().sum()),0)
        self.assertEqual(float(logits.grad[:,:,12:,:].abs().sum()),0)
        with torch.no_grad():
            changed=logits.detach().clone();changed[:,:,12:,:]=100;changed[:,:,:,12:]=-100
            again,_=complete_loss(changed,target,valid,(12,12))
        self.assertAlmostEqual(float(loss.detach()),float(again),places=7)

if __name__=='__main__':unittest.main()
