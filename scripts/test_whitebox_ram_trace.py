"""Acceptance checks against actual diagnostic artifacts, no training."""
import json
import math
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'outputs/whitebox_20260922'


class RealTraceAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[json.loads(x) for x in (ROOT/'reference/TRAINING_TRACE.jsonl').read_text().splitlines()]
        cls.audit=json.loads((ROOT/'reference/PARAMETER_AUDIT.json').read_text())

    def test_complete_real_update_accounting(self):
        self.assertEqual(len(self.rows),12)
        self.assertEqual(self.rows[-1]['successful_optimizer_steps'],12)
        self.assertEqual(len({r['case'] for r in self.rows}),self.rows[-1]['unique_cases'])
        for r in self.rows:
            self.assertFalse(r['AMP_skipped'])
            self.assertTrue(all(v['update_norm']>0 for v in r['actual_parameter_updates'].values()))
            for v in r['actual_parameter_updates'].values():
                self.assertEqual(v['optimizer_step_range'],[r['step'],r['step']])

    def test_loss_component_reconstruction(self):
        for r in self.rows:
            weighted=sum(t['weighted'] for t in r['loss_terms'].values())
            self.assertAlmostEqual(weighted,r['loss'],places=7)

    def test_observer_and_optimizer_integrity(self):
        self.assertTrue(self.audit['optimizer_coverage_exact'])
        eq=self.rows[0]['observation_equivalence']
        self.assertEqual(eq['loss_abs'],0.)
        self.assertEqual(eq['gradient_max_abs'],0.)
        self.assertTrue(eq['none_pattern_equal'])
        for r in self.rows:
            for v in r['gradient_optimizer_receives'].values():
                self.assertEqual(v['nonfinite_elements'],0)
                self.assertTrue(math.isfinite(v['norm']))
            self.assertEqual(r['explicit_o_weight']['norm'],0.)

    def test_matched_lr_intervention(self):
        other=[json.loads(x) for x in (ROOT/'variant_tail_lr/TRAINING_TRACE.jsonl').read_text().splitlines()]
        self.assertEqual(len(other),12)
        self.assertEqual(other[-1]['successful_optimizer_steps'],12)
        self.assertEqual([r['batch_sha256'] for r in self.rows],[r['batch_sha256'] for r in other])
        self.assertEqual(self.rows[0]['loss'],other[0]['loss'])
        self.assertEqual(self.rows[0]['gradient_before_clip'],other[0]['gradient_before_clip'])
        for group,v in self.rows[0]['actual_parameter_updates'].items():
            w=other[0]['actual_parameter_updates'][group]
            if group.startswith('model.'):
                self.assertEqual(v['update_norm'],w['update_norm'])
            else:
                self.assertGreater(w['update_norm'],0.)
                self.assertLess(w['update_norm'],v['update_norm']/100)

    def test_complete_identical_validation_cohort(self):
        ref=json.loads((ROOT/'reference/FINAL_VALIDATION.json').read_text())
        var=json.loads((ROOT/'variant_tail_lr/FINAL_VALIDATION.json').read_text())
        self.assertEqual(ref['cases'],var['cases'])
        self.assertEqual(len(ref['cases']),69)
        for arm in [ref,var]:
            for stage in ['step0','step1','step2']:
                m=arm['official'][stage]
                self.assertEqual(m['overall_instance_metrics']['n'],966)
                self.assertEqual(m['overlap_region_metrics']['n'],69)
                self.assertEqual(m['overlap_pair_intersection_metrics']['n'],1004)
        # Prespecified short-run mitigation gate; not a claim of beating mature M0.
        for metric in ['overall_dsc','overlap_dsc','overlap_nsd_2px']:
            self.assertGreater(var['flat']['step2'][metric],ref['flat']['step2'][metric])


if __name__=='__main__':unittest.main()
