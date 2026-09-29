#!/usr/bin/env python3
"""Summarize read-only fixed-H0 source-head response and error tradeoffs."""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    path = root / 'response_v1' / 'RESULT.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    assert result['status'] == 'COMPLETE' and result['optimizer_updates'] == 0
    arms = result['by_split_counts']['arms']
    a = arms['OHEAD_ZERO']
    b = arms['OHEAD_REAL']
    keys = ('h0_member_recovered', 'member_fn_all', 'member_fn_outside_h0',
            'fp_all_valid', 'unique_fn_all', 'unique_global_overlap_fn_recovered')
    aggregate = {}
    for subset in ('train-update', 'train-heldout', 'val'):
        aggregate[subset] = {}
        for state in ('source', 'step1', 'step2'):
            zero = a[subset]['states'][state]
            real = b[subset]['states'][state]
            aggregate[subset][state] = {k: {
                'zero': zero[k], 'real': real[k], 'delta': real[k] - zero[k]
            } for k in keys}

    trace = [json.loads(line) for line in (root / 'response_v1' / 'TRAINING_TRACE_OR_EQUIV.jsonl').open(encoding='utf-8')]
    assert len(trace) == 291
    by_arm = {arm: {} for arm in ('OHEAD_ZERO', 'OHEAD_REAL')}
    for row in trace:
        if row['record_type'] == 'arm_case_response':
            by_arm[row['arm']][(row['subset'], row['case'])] = row
    assert set(by_arm['OHEAD_ZERO']) == set(by_arm['OHEAD_REAL'])
    cases = []
    for subset, case in sorted(by_arm['OHEAD_ZERO']):
        z = by_arm['OHEAD_ZERO'][(subset, case)]
        r = by_arm['OHEAD_REAL'][(subset, case)]
        row = {'subset': subset, 'case': case, 'h0_members': z['h0']['member_pixels']}
        for state in ('source', 'step1', 'step2'):
            zs = z['states'][state]
            rs = r['states'][state]
            for key in ('h0_member_recovered', 'member_fn_outside_h0', 'fp_all_valid'):
                row[f'{state}_{key}_zero'] = zs[key]
                row[f'{state}_{key}_real'] = rs[key]
                row[f'{state}_{key}_delta'] = rs[key] - zs[key]
            row[f'{state}_h0_logit_mean_delta'] = rs['logit_h0']['mean'] - zs['logit_h0']['mean']
        cases.append(row)
    with (root / 'PER_CASE_H0_RESPONSE_DIFF.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    pooled = {}
    for subset in ('train-update', 'train-heldout', 'val'):
        rows = [x for x in cases if x['subset'] == subset]
        count = sum(x['h0_members'] for x in rows)
        pooled[subset] = {'cases': len(rows), 'h0_members': count}
        for state in ('source', 'step1', 'step2'):
            pooled[subset][f'{state}_h0_logit_mean_delta'] = sum(
                x[f'{state}_h0_logit_mean_delta'] * x['h0_members'] for x in rows
            ) / count
            changes = [x[f'{state}_h0_member_recovered_delta'] for x in rows]
            pooled[subset][f'{state}_h0_case_positive'] = sum(x > 0 for x in changes)
            pooled[subset][f'{state}_h0_case_negative'] = sum(x < 0 for x in changes)
            pooled[subset][f'{state}_h0_case_unchanged'] = sum(x == 0 for x in changes)

    summary = {
        'status': 'COMPLETE', 'test_used': False, 'read_only': True,
        'diagnostic_label': result['diagnostic_label'],
        'expected_trace_rows': 291, 'actual_trace_rows': len(trace),
        'aggregate': aggregate, 'pooled_h0': pooled,
        'limitations': ['count equality does not prove identical recovered coordinates',
                        'read-only response is diagnostic; official metrics come from FINAL_VALIDATION.json'],
    }
    (root / 'RESPONSE_COMPARISON.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pooled_h0': pooled, 'val_final': aggregate['val']['step2']}, indent=2))


if __name__ == '__main__':
    main()
