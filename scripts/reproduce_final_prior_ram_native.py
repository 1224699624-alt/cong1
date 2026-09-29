"""Reproduce the frozen RAM native checkpoint on all 69 validation cases."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from r351_data import AlignedRamDataset
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r351_ram_dual_prior import collect_native, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    checkpoint = a.root / 'outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth'
    expected = '88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9'
    assert sha256(checkpoint) == expected
    dataset = a.root / 'data/remote_variants/RAM-W600'
    ds = AlignedRamDataset(dataset, a.root / 'outputs/priors/r323_ram_native_r317_single_seed', 'val')
    assert len(ds) == 69
    torch.manual_seed(3511)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(2)
    device = torch.device('cuda')
    model = NnUNetMultiLabelPrior().to(device)
    payload = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(payload['model'], strict=True)
    model.requires_grad_(False).eval()
    result = collect_native(model, DataLoader(ds, batch_size=1, shuffle=False, num_workers=2),
                            device, top_training_pairs(dataset, 15), .25)
    report = {'source_checkpoint': str(checkpoint), 'source_checkpoint_sha256': expected,
              'source_epoch': payload['row']['epoch'], 'test_used': False,
              'split': 'val', 'num_images': 69, 'native_baseline': result,
              'evaluator_sha256': sha256(a.root / 'scripts/evaluate_r329_ram_official_metrics_val.py')}
    old_path = Path('/root/r351_workspace/ram_pilot/result.json')
    if old_path.exists():
        old = json.loads(old_path.read_text())['native_baseline']['flat']
        report['deltas_to_historical'] = {k: v - old[k] for k, v in result['flat'].items()
                                           if v is not None and old.get(k) is not None}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'metrics': result['official']['overall_instance_metrics'],
                      'deltas': report.get('deltas_to_historical')}), flush=True)


if __name__ == '__main__':
    main()
