import unittest
import torch
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner
from train_r332_ram_joint_iterative_refinement import iterative_refine
from r332_v31_port import R332RefinerWithO,load_mature_with_zero_o,iterative_refine_o

class PortTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(332);torch.set_num_threads(1)
    def test_zero_o_preserves_all_steps_and_mature_heads(self):
        old=InstanceCompletionRefiner().eval()
        with torch.no_grad():
            for p in old.parameters():p.normal_(0,.02)
            old.gate.bias.fill_(-1.3);old.residual.bias.fill_(.04)
        new=R332RefinerWithO().eval();report=load_mature_with_zero_o(new,old.state_dict())
        image=torch.randn(1,1,32,32);logits=torch.randn(1,14,32,32)
        with torch.no_grad():a,_=iterative_refine(image,logits,old,2);b,_=iterative_refine_o(image,logits,new,2,torch.zeros_like(image))
        self.assertTrue(report['mature_residual_exact'] and report['mature_gate_exact'])
        self.assertEqual(report['new_o_weight_nonzero'],0)
        for x,y in zip(a,b):self.assertLessEqual(float((x-y).abs().max()),1e-6)
    def test_nonzero_o_can_reach_output_after_o_weights_enabled(self):
        old=InstanceCompletionRefiner()
        with torch.no_grad():old.residual.weight.fill_(.02);old.gate.weight.fill_(.01)
        new=R332RefinerWithO();load_mature_with_zero_o(new,old.state_dict())
        with torch.no_grad():new.enc1.net[0].weight[:,6].fill_(.01)
        image=torch.randn(1,1,16,16);logits=torch.randn(1,14,16,16)
        a=iterative_refine_o(image,logits,new,1,torch.zeros_like(image))[0][-1]
        b=iterative_refine_o(image,logits,new,1,torch.ones_like(image))[0][-1]
        self.assertGreater(float((a-b).detach().abs().max()),0.)
if __name__=='__main__':unittest.main()
