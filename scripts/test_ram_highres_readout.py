"""Focused synthetic invariants for the frozen RAM P0/P1 readout runner."""
import unittest
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_ram_highres_readout import (
    B_NAMES, FeatureTransform, feature_at, make_q, threshold_for_fpr,
    fit_probes,
    evaluate_natural, Probe,
)


class HighresReadoutTests(unittest.TestCase):
    def test_q_uses_only_predictions_and_original_valid(self):
        source=np.full((14,4,5),-7.0,dtype=np.float32)
        final=np.full((14,4,5),-7.0,dtype=np.float32)
        final[0,1,2]=3.0
        valid=np.ones((4,5),dtype=bool)
        valid[0,0]=False
        q=make_q(source,final,valid)
        self.assertTrue(q[1,1,2])
        self.assertFalse(q[0,1,2])
        self.assertFalse(q[:,0,0].any())
        self.assertFalse(q[:,3,4].any())

    def test_feature_shapes_and_member_condition(self):
        zeros=lambda shape:np.zeros(shape,dtype=np.float32)
        record={name:zeros((14,4,5)) for name in ('source','step1','step2')}
        record.update({'image':zeros((4,5)),'gradient':zeros((4,5)),
                       'backbone_h':zeros((32,4,5)),
                       'refiner_h':zeros((14,12,4,5))})
        record['step1'][0,1,2]=2.0
        record['refiner_h'][0,3,1,2]=5.0
        record['backbone_h'][3,1,2]=7.0
        b,h=feature_at(record,np.array([0]),np.array([1]),np.array([2]))
        self.assertEqual(b.shape,(1,len(B_NAMES)))
        self.assertEqual(h.shape,(1,32))
        self.assertAlmostEqual(float(b[0,15+3]),5.0)
        self.assertAlmostEqual(float(h[0,3]),7.0)
        self.assertAlmostEqual(float(b[0,2]),1/(1+np.exp(-2)),places=6)

    def test_capacity_control_same_input_width(self):
        arrays={'B':np.arange(5*len(B_NAMES),dtype=np.float32).reshape(5,-1),
                'H':np.arange(5*32,dtype=np.float32).reshape(5,32)}
        transform=FeatureTransform(arrays,4421)
        member=np.array([0,1,2,3,13])
        x0=transform.apply(arrays['B'],arrays['H'],member,'P0')
        x1=transform.apply(arrays['B'],arrays['H'],member,'P1')
        self.assertEqual(x0.shape,x1.shape)
        self.assertEqual(x0.shape[1],transform.input_dim)
        np.testing.assert_array_equal(x0[:,-14:],x1[:,-14:])
        np.testing.assert_array_equal(x0[:,:len(B_NAMES)],x1[:,:len(B_NAMES)])

    def test_low_fpr_threshold_is_conservative_with_ties(self):
        score=np.array([.9,.9,.8,.7,.2],dtype=np.float32)
        weights=np.ones(5,dtype=np.float32)
        threshold=threshold_for_fpr(score,weights,.2)
        self.assertLessEqual(float((score>=threshold).mean()),.2+1e-12)

    def test_paired_probe_fit_same_steps_on_synthetic_cache(self):
        with tempfile.TemporaryDirectory(prefix='ram_highres_probe_test_') as folder:
            root=Path(folder)
            rng=np.random.default_rng(17)
            n=1024
            np.savez_compressed(root/'SAMPLES_probe_fit.npz',
                B=rng.normal(size=(n,len(B_NAMES))).astype(np.float32),
                H=rng.normal(size=(n,32)).astype(np.float32),
                label=np.r_[np.zeros(n//2),np.ones(n//2)].astype(np.int8),
                member=(np.arange(n)%14).astype(np.int16),
                h0=np.zeros(n,dtype=np.int8),
                kind=(np.arange(n)%3).astype(np.int8))
            lock_path=root/'LOCK.json'
            lock_path.write_text('{}',encoding='utf-8')
            lock={'_path':str(lock_path),'seed':4421,'hidden_width':32,
                  'epochs':1,'batch_size':256,'max_optimizer_steps_per_arm':4,
                  'learning_rate':0.001,'weight_decay':0.0001,
                  'C0_sha256':'ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384'}
            _,_,result=fit_probes(lock,root,torch.device('cpu'))
            self.assertEqual(result['P0']['steps'],4)
            self.assertEqual(result['P1']['steps'],4)
            self.assertTrue((root/'PROBE_TRAINING_TRACE.jsonl').is_file())

    def test_natural_q_evaluation_and_threshold_on_tiny_case(self):
        class FakeDataset:
            def __init__(self,*_args,**_kwargs):
                self.mask_files=[Path('tiny.npy')]
        with tempfile.TemporaryDirectory(prefix='ram_highres_natural_test_') as folder:
            root=Path(folder)
            split=root/'split.json'
            split.write_text('{"groups":{"probe_calibration":["tiny"]},"val_cases":[]}',encoding='utf-8')
            rng=np.random.default_rng(19)
            transform=FeatureTransform({'B':rng.normal(size=(12,len(B_NAMES))).astype(np.float32),
                                        'H':rng.normal(size=(12,32)).astype(np.float32)},4421)
            probes={'P0':Probe(transform.input_dim,32).eval(),
                    'P1':Probe(transform.input_dim,32).eval()}
            rec={name:np.zeros((14,4,5),dtype=np.float32) for name in ('source','step1','step2')}
            rec.update({'image':np.zeros((4,5),dtype=np.float32),
                        'gradient':np.zeros((4,5),dtype=np.float32),
                        'backbone_h':np.zeros((32,4,5),dtype=np.float32),
                        'refiner_h':np.zeros((14,12,4,5),dtype=np.float32)})
            q=np.zeros((14,4,5),dtype=bool)
            q[0,1,1]=True;q[0,1,2]=True;q[1,2,1]=True;q[1,2,2]=True
            target=np.zeros_like(q);target[0,1,1]=True;target[1,2,1]=True
            h0=np.zeros_like(q);h0[0,1,1]=True
            lock={'split_manifest':str(split),'dataset':'unused',
                  'oracle_h0_root':'unused','seed':4421,
                  'natural_eval_cap_per_case':0,'score_chunk':2,
                  'score_batch':2,'low_fpr_alpha':.001,
                  'disk_free_floor_bytes':0}
            with patch('run_ram_highres_readout.legacy.NativeWristDataset',FakeDataset), \
                 patch('run_ram_highres_readout.load_case_and_capture',
                       return_value=(rec,target,h0,q,{'case':'tiny'})):
                result,thresholds=evaluate_natural(lock,root,None,None,
                    torch.device('cpu'),transform,probes,'probe_calibration',None,
                    float('inf'))
            self.assertEqual(result['population_exact']['q_members'],4)
            self.assertEqual(result['population_exact']['q_positive'],2)
            self.assertEqual(result['population_exact']['h0_in_q'],1)
            self.assertEqual(set(thresholds),{'P0','P1'})


if __name__=='__main__':
    unittest.main()
