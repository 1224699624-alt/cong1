"""Validate completed real artifacts without requiring a positive outcome."""
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'outputs/whitebox_adam_20260922'
def read(p):return json.loads(p.read_text())


class RealAdamPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traces={arm:[json.loads(x) for x in (ROOT/arm/'TRAINING_TRACE.jsonl').read_text().splitlines()] for arm in ['fresh_adam','restored_adam']}

    def test_same_input_budget_and_lr(self):
        a,b=self.traces.values()
        self.assertEqual(len(a),100);self.assertEqual(len(b),100)
        self.assertEqual([x['batch_sha256'] for x in a],[x['batch_sha256'] for x in b])
        self.assertEqual(a[0]['loss'],b[0]['loss'])
        for rows in [a,b]:
            self.assertEqual(rows[-1]['successful_optimizer_steps'],100)
            for row in rows:
                self.assertEqual(row['lr'],[2.1912418526906397e-7,7e-6])
                self.assertFalse(row['AMP_skipped'])
                self.assertEqual(row['explicit_o_weight']['norm'],0.)

    def test_historical_step_counters_preserved(self):
        for arm,rows in self.traces.items():
            for row in rows:
                for name,group in row['actual_parameter_updates'].items():
                    offset=0 if arm=='fresh_adam' else (12746 if name.startswith('refiner.') else 11046)
                    self.assertEqual(group['optimizer_step_range'],[offset+row['step']]*2)
        mapping=read(ROOT/'restored_adam/ADAM_STATE_MAPPING.json')
        self.assertEqual(mapping['status'],'PASS');self.assertEqual(len(mapping['parameters']),62)

    def test_shared_gradient_shadow(self):
        r=read(ROOT/'SHARED_GRADIENT_PROBE.json')
        self.assertEqual(r['status'],'PASS');self.assertEqual(r['source_model_optimizer_steps'],0)
        self.assertEqual(r['shadow_optimizer_steps'],2)
        self.assertEqual(r['arms']['fresh']['gradient_sha256'],r['arms']['restored']['gradient_sha256'])

    def test_same_standard_validation(self):
        for filename in ['VALIDATION_step0012.json','FINAL_VALIDATION.json']:
            a,b=[read(ROOT/arm/filename) for arm in self.traces]
            self.assertEqual(a['cases'],b['cases']);self.assertEqual(len(a['cases']),69)
            for v in [a,b]:
                self.assertEqual(v['official']['step2']['overall_instance_metrics']['n'],966)
                self.assertEqual(v['official']['step2']['overlap_pair_intersection_metrics']['n'],1004)
        for arm in self.traces:self.assertEqual(read(ROOT/arm/'RESULT.json')['status'],'COMPLETE')


if __name__=='__main__':unittest.main()
