"""Require the new O=zero control to reproduce the matched old F0S1 reference."""

import argparse
import json
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--zero', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    reference = read(args.reference / 'run_manifest.json')
    zero = read(args.zero / 'run_manifest.json')
    if any(x['status'] != 'COMPLETE' or x['optimizer_steps'] != 384 for x in (reference, zero)):
        raise ValueError('reference or zero arm incomplete')
    if reference['checkpoint_sha256'] != zero['checkpoint_sha256'] or \
       reference['schedule_sha256'] != zero['schedule_sha256']:
        raise ValueError('C0 or schedule differs')
    a = [json.loads(x) for x in (args.reference / 'TRAINING_TRACE.jsonl').open(encoding='utf-8')]
    b = [json.loads(x) for x in (args.zero / 'TRAINING_TRACE.jsonl').open(encoding='utf-8')]
    if len(a) != 384 or len(b) != 384:
        raise ValueError('trace length differs')
    for old, new in zip(a, b):
        for key in ('step', 'case', 'batch_sha256', 'fixed_indices', 'h0_exposure', 'denominator_c0'):
            if old[key] != new[key]:
                raise ValueError(f'{key} differs at step {old["step"]}')
        if old['AMP'] != new['AMP']:
            raise ValueError(f'AMP differs at step {old["step"]}')
        for key in ('native_loss', 'extra_source_loss', 'total_loss'):
            if abs(old[key] - new[key]) > 1e-6:
                raise ValueError(f'{key} differs at step {old["step"]}')
    old_val = read(args.reference / 'FINAL_VALIDATION.json')
    new_val = read(args.zero / 'FINAL_VALIDATION.json')
    if old_val['status'] != 'COMPLETE' or new_val['status'] != 'COMPLETE' or \
       old_val['result']['cases'] != new_val['result']['cases']:
        raise ValueError('val scope differs')
    old_metrics = old_val['result']['flat']['step2']
    new_metrics = new_val['result']['flat']['step2']
    differences = {key: new_metrics[key] - old_metrics[key]
                   for key in old_metrics if isinstance(old_metrics[key], (float, int))}
    old_state = torch.load(args.reference / 'diagnostic_final.pth', map_location='cpu', weights_only=False)
    new_state = torch.load(args.zero / 'diagnostic_final.pth', map_location='cpu', weights_only=False)
    core_differences = {}
    for part in ('model', 'refiner'):
        for key, value in old_state[part].items():
            other = new_state[part].get(key)
            if other is None or value.shape != other.shape:
                raise ValueError(f'{part}.{key} missing or wrong shape')
            difference = float((other.float() - value.float()).abs().max())
            if difference:
                core_differences[f'{part}.{key}'] = difference
    adapter = new_state['model']['o_source_adapter.weight']
    result = {
        'status': 'PASS' if not core_differences and all(abs(value) <= 1e-8 for value in differences.values())
                  and int(torch.count_nonzero(adapter)) == 0 else 'FAIL',
        'same_C0_schedule_batches_members_losses_AMP': True,
        'reference_updates': 384, 'zero_updates': 384,
        'core_tensor_nonzero_difference_count': len(core_differences),
        'core_tensor_max_abs_difference': max(core_differences.values(), default=0.0),
        'adapter_nonzero_weights': int(torch.count_nonzero(adapter)),
        'standard_metric_deltas': differences,
        'test_used': False,
    }
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise RuntimeError('zero-O parity failed; stop before real O arm')


if __name__ == '__main__':
    main()
