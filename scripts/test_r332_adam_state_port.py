"""Small synthetic CPU tensors only: no dataset or real model training."""
import copy
import unittest
import torch
from r332_adam_state_port import adapt_tensor, attach_named_state


class AdamPortTests(unittest.TestCase):
    def fixture(self):
        names=['refiner.enc1.net.0.weight','model.backbone.decoder.seg_layers.4.weight']
        old=[torch.nn.Parameter(torch.ones(12,6,3,3)),torch.nn.Parameter(torch.ones(15,32,1,1))]
        new=[torch.nn.Parameter(adapt_tensor(n,p.detach(),s)[0]) for n,p,s in zip(names,old,[(12,7,3,3),(14,32,1,1)])]
        def opt(ps):return torch.optim.AdamW([{'params':[ps[0]],'lr':2.1912418526906397e-7},{'params':[ps[1]],'lr':7e-6}],weight_decay=1e-4)
        original=opt(old);dest=opt(new)
        for p in old:original.state[p]={'step':torch.tensor(10.),'exp_avg':torch.full_like(p,.03),'exp_avg_sq':torch.full_like(p,.02)}
        saved=copy.deepcopy(original.state_dict())
        for g in saved['param_groups']:g['lr']=0.
        source_groups=[[(names[i],old[i])] for i in range(2)];dest_groups=[[(names[i],new[i])] for i in range(2)]
        return old,new,original,dest,saved,source_groups,dest_groups

    def test_moments_and_lrs(self):
        old,new,original,dest,saved,sg,dg=self.fixture()
        rows=attach_named_state(dest,dg,sg,saved)
        self.assertEqual(len(rows),2)
        self.assertEqual(dest.param_groups[0]['lr'],2.1912418526906397e-7)
        self.assertTrue(torch.equal(dest.state[new[0]]['exp_avg'][:,:6],original.state[old[0]]['exp_avg']))
        self.assertEqual(torch.count_nonzero(dest.state[new[0]]['exp_avg'][:,6]).item(),0)
        self.assertTrue(torch.equal(dest.state[new[1]]['exp_avg_sq'],original.state[old[1]]['exp_avg_sq'][:14]))

    def test_real_adam_step_equivalence_on_synthetic_tensors(self):
        old,new,original,dest,saved,sg,dg=self.fixture();attach_named_state(dest,dg,sg,saved)
        for p in old:p.grad=torch.full_like(p,.02)
        new[0].grad=torch.zeros_like(new[0]);new[0].grad[:,:6]=old[0].grad
        new[1].grad=old[1].grad[:14].clone()
        original.step();dest.step()
        self.assertTrue(torch.equal(new[0][:,:6],old[0]))
        self.assertTrue(torch.equal(new[1],old[1][:14]))
        self.assertEqual(torch.count_nonzero(new[0][:,6]).item(),0)
        self.assertEqual(float(dest.state[new[0]]['step']),11.)

    def test_reject_unknown_shape_and_incomplete_state(self):
        with self.assertRaises(ValueError):adapt_tensor('refiner.some.weight',torch.zeros(2,2),(2,3))
        old,new,original,dest,saved,sg,dg=self.fixture()
        saved['state'][0]['exp_avg']=torch.zeros(1)
        with self.assertRaises(ValueError):attach_named_state(dest,dg,sg,saved)
        self.assertEqual(len(dest.state),0)

    def test_names_not_destination_position(self):
        # Two same-shaped parameters intentionally reversed in the new group.
        a=torch.nn.Parameter(torch.ones(2));b=torch.nn.Parameter(torch.ones(2));c=torch.nn.Parameter(torch.ones(2))
        dest=torch.optim.AdamW([{'params':[b,a],'lr':.01},{'params':[c],'lr':.02}])
        saved=copy.deepcopy(dest.state_dict())
        for idx in [0,1,2]:saved['state'][idx]={'step':torch.tensor(4.),'exp_avg':torch.full((2,),float(idx+1)),'exp_avg_sq':torch.ones(2)}
        attach_named_state(dest,[[('b',b),('a',a)],[('c',c)]],[[('a',a),('b',b)],[('c',c)]],saved)
        self.assertTrue(torch.equal(dest.state[a]['exp_avg'],torch.ones(2)))
        self.assertTrue(torch.equal(dest.state[b]['exp_avg'],torch.full((2,),2.)))


if __name__=='__main__':unittest.main()
