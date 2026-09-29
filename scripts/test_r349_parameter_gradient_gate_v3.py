import hashlib,json,sys,unittest
from pathlib import Path
import torch
from r349_parameter_gradient_gate import route_parameter_gradients
from r349_parameter_gradient_gate_v3 import route_parameter_gradients_v3,assign_gradients_v3


def probe():
    result={}
    for name,fn in [('legacy',route_parameter_gradients),('v3',route_parameter_gradients_v3)]:
        x=torch.tensor(0.,requires_grad=True);y=torch.tensor(0.,requires_grad=True)
        kwargs={'max_aux_ratio':10.}
        if name=='v3':kwargs['max_merged_aux_ratio']=None
        g,stats=fn([x,y],x+y,[('partial',-x,1.)],**kwargs)
        aux=[float(t)-1 for t in g]
        result[name]={'combined':[float(t) for t in g],'projected_aux':aux,'aux_dot_base':sum(aux)}
    return result


class RoutingTests(unittest.TestCase):
    def test_partial_none_exact_projection(self):
        r=probe();self.assertLess(r['legacy']['aux_dot_base'],-.49)
        for a,b in zip(r['v3']['projected_aux'],[-.5,.5]):self.assertAlmostEqual(a,b,places=6)

    def test_unused_frozen_and_optimizer(self):
        x=torch.nn.Parameter(torch.tensor(1.));unused=torch.nn.Parameter(torch.tensor(2.));frozen=torch.nn.Parameter(torch.tensor(3.),requires_grad=False)
        g,_=route_parameter_gradients_v3([frozen,x,unused],x*x,[('zero',unused*2,0.)])
        self.assertIsNone(g[1]);assign_gradients_v3([frozen,x,unused],g)
        opt=torch.optim.AdamW([frozen,x,unused],lr=.01,weight_decay=1.)
        opt.step();self.assertEqual(float(unused.detach()),2.);self.assertEqual(float(frozen),3.)
        self.assertNotIn(unused,opt.state)

    def test_zero_and_disconnected_losses(self):
        for mode in ['constant','zero','unrelated']:
            x=torch.nn.Parameter(torch.tensor(1.));y=torch.tensor(1.,requires_grad=True)
            base={'constant':torch.tensor(0.),'zero':x*0,'unrelated':y*2}[mode]
            g,rows=route_parameter_gradients_v3([x],base,[('aux',x,1.)])
            self.assertEqual(rows[-1]['merged_aux_norm'],0.)
            if mode!='zero':self.assertIsNone(g[0])

    def test_multi_aux_caps_and_direction(self):
        x=torch.nn.Parameter(torch.tensor(1.));y=torch.nn.Parameter(torch.tensor(1.))
        g,rows=route_parameter_gradients_v3([x,y],x+y,[('seam',10*x,1.),('overlap',10*y,1.)],max_aux_ratio=.15,max_merged_aux_ratio=.1)
        self.assertLessEqual(rows[-1]['merged_aux_base_ratio'],.100001)
        self.assertGreaterEqual(rows[-1]['merged_aux_dot_base'],-1e-6)
        for row in rows[:-1]:self.assertLessEqual(row['applied_norm_before_merged_cap']/row['base_norm'],.150001)

    def test_positive_negative_zero_and_no_aux(self):
        for factor in [-1.,0.,1.]:
            x=torch.nn.Parameter(torch.tensor(1.))
            g,_=route_parameter_gradients_v3([x],x,[('aux',factor*x,1.)])
            self.assertTrue(torch.isfinite(g[0]));self.assertGreaterEqual(float(g[0]),.99999)
        x=torch.nn.Parameter(torch.tensor(1.));g,_=route_parameter_gradients_v3([x],x,[])
        self.assertEqual(float(g[0]),1.)

    def test_negative_weight(self):
        x=torch.nn.Parameter(torch.tensor(1.))
        with self.assertRaises(ValueError):route_parameter_gradients_v3([x],x,[('bad',x,-1.)])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RoutingTests))
    root=Path(__file__).resolve().parents[1];out=root/'outputs/v3_round2';out.mkdir(parents=True,exist_ok=True)
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','kind':'CPU_SYNTHETIC_NOT_MODEL_EXPERIMENT',
            'tests_run':result.testsRun,'probe':probe(),'torch':torch.__version__,
            'command':f'{sys.executable} scripts/test_r349_parameter_gradient_gate_v3.py',
            'version_sha256':hashlib.sha256((root/'scripts/r349_parameter_gradient_gate_v3.py').read_bytes()).hexdigest()}
    (out/'GRADIENT_ROUTING_REGRESSION.json').write_text(json.dumps(report,indent=2))
    sys.exit(not result.wasSuccessful())
