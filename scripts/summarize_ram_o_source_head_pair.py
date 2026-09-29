#!/usr/bin/env python3
"""Summarize the locked zero-O/real-O RAM source-head comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = (
    'overall_dsc', 'overall_nsd_2px', 'overall_msd_px',
    'overlap_dsc', 'overlap_nsd_2px', 'overlap_msd_px',
    'pair_dsc', 'pair_nsd_2px', 'pair_msd_px', 'pair_msd_fail_rate',
)
CASE_FIELDS = {
    'overall': 'overall_instance_metrics',
    'overlap': 'overlap_region_metrics',
    'pair': 'overlap_pair_intersection_metrics',
}


def load_trace(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    zero = root / 'OHEAD_ZERO_v1'
    real = root / 'OHEAD_REAL_v1'
    zval = json.loads((zero / 'FINAL_VALIDATION.json').read_text(encoding='utf-8'))
    rval = json.loads((real / 'FINAL_VALIDATION.json').read_text(encoding='utf-8'))
    ztrace = load_trace(zero / 'TRAINING_TRACE.jsonl')
    rtrace = load_trace(real / 'TRAINING_TRACE.jsonl')
    run_lock = json.loads((root / 'O_SOURCE_HEAD_RUN_LOCK_v1.json').read_text(encoding='utf-8'))
    assert len(ztrace) == len(rtrace) == 384
    matching = {
        'step': all(a['step'] == b['step'] for a, b in zip(ztrace, rtrace)),
        'case': all(a['case'] == b['case'] for a, b in zip(ztrace, rtrace)),
        'batch_sha256': all(a['batch_sha256'] == b['batch_sha256'] for a, b in zip(ztrace, rtrace)),
        'fixed_indices': all(a['fixed_indices'] == b['fixed_indices'] for a, b in zip(ztrace, rtrace)),
        'lr': all(a['lr'] == b['lr'] for a, b in zip(ztrace, rtrace)),
        'amp_no_skip': all(not a['AMP']['skipped'] and not b['AMP']['skipped'] for a, b in zip(ztrace, rtrace)),
    }
    if not all(matching.values()):
        raise RuntimeError(f'matched protocol failed: {matching}')

    zflat = zval['result']['flat']['step2']
    rflat = rval['result']['flat']['step2']
    rows = []
    for key in METRICS:
        rows.append({'dataset': 'RAM train425/val69',
                     'start_C0_sha256': run_lock['C0']['sha256'],
                     'updates_per_arm': 384,
                     'matched_batch_sha256': matching['batch_sha256'],
                     'only_variable': 'source-head O zero versus frozen real',
                     'metric': key, 'OHEAD_ZERO': zflat[key], 'OHEAD_REAL': rflat[key],
                     'delta_real_minus_zero': rflat[key] - zflat[key],
                     'direction': 'lower_better' if 'msd' in key else 'higher_better'})
    with (root / 'RUN_CONFIG_AND_RESULT_DIFF.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    zcases = zval['result']['per_case_official']
    rcases = rval['result']['per_case_official']
    assert set(zcases) == set(rcases) and len(zcases) == 69
    case_rows = []
    for case in sorted(zcases):
        row = {'case': case}
        for group, field in CASE_FIELDS.items():
            for metric in ('dsc', 'nsd_2px', 'msd_px'):
                a = zcases[case]['step2'][field][metric]
                b = rcases[case]['step2'][field][metric]
                row[f'{group}_{metric}_zero'] = a
                row[f'{group}_{metric}_real'] = b
                row[f'{group}_{metric}_delta'] = b - a
        case_rows.append(row)
    with (root / 'PER_CASE_STANDARD_DIFF.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0]))
        writer.writeheader()
        writer.writerows(case_rows)

    case_summary = {}
    for group in CASE_FIELDS:
        for metric in ('dsc', 'nsd_2px', 'msd_px'):
            key = f'{group}_{metric}_delta'
            x = np.asarray([row[key] for row in case_rows], dtype=float)
            case_summary[key] = {
                'mean': float(x.mean()), 'median': float(np.median(x)),
                'positive': int((x > 0).sum()), 'negative': int((x < 0).sum()),
                'zero': int((x == 0).sum()),
            }

    def trace_stats(rows: list[dict]) -> dict:
        final = rows[-1]
        adapter_delta = np.array([r['parameter_updates']['o_source_adapter']['delta_theta_l2'] for r in rows])
        adapter_grad = np.array([r['clip']['before']['o_source_adapter']['l2'] for r in rows])
        adapter_response = np.array([r['o_source_flow']['adapter_response_l2'] for r in rows])
        return {
            'steps': len(rows),
            'first_case': rows[0]['case'], 'last_case': final['case'],
            'successful_optimizer_steps': final['optimizer_steps']['successful'],
            'adapter_update_nonzero_steps': int((adapter_delta > 0).sum()),
            'adapter_grad_nonzero_steps': int((adapter_grad > 0).sum()),
            'adapter_response_nonzero_steps': int((adapter_response > 0).sum()),
            'adapter_response_l2_last': float(adapter_response[-1]),
            'adapter_cumulative_offset_l2': final['parameter_updates']['o_source_adapter']['cumulative_offset_l2'],
            'existing_decoder_cumulative_offset_l2': final['parameter_updates']['decoder_existing']['cumulative_offset_l2'],
            'refiner_cumulative_offset_l2': final['parameter_updates']['refiner']['cumulative_offset_l2'],
            'adapter_lr_last': final['lr']['o_source_adapter'],
            'peak_gpu_bytes': json.loads((zero if rows is ztrace else real).joinpath('run_manifest.json').read_text())['peak_gpu_bytes'],
        }
    summary = {
        'status': 'COMPLETE', 'test_used': False, 'RAM': 'train425/val69',
        'matched_protocol': matching,
        'official_step2': {'zero': zflat, 'real': rflat, 'delta': {r['metric']: r['delta_real_minus_zero'] for r in rows}},
        'per_case': case_summary,
        'training': {'zero': trace_stats(ztrace), 'real': trace_stats(rtrace)},
    }
    (root / 'PAIR_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'matching': matching, 'metrics': rows, 'training': summary['training']}, indent=2))


if __name__ == '__main__':
    main()
