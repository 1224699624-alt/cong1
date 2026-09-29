"""Read-only same-batch final H0 objective response across the first AdamW step."""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ram_h0_final_target as runner


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--response-code', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    sys.path.insert(0, str(a.response_code.resolve()))
    response = importlib.import_module('audit_ram_h0_final_target_response')
    lock_path = a.lock.resolve()
    lock = json.loads(lock_path.read_text())
    steps, _, _ = runner._load_schedule(lock, lock_path)
    batch, batch_hash = runner._load_step_payload(steps[0])
    device = torch.device('cuda')
    _, model, refiner, generator, optimizer, scaler, _, _, _, _ = \
        runner._prepare_components(lock, lock_path, 'SRC_F0S1', device)
    image = batch['image'].unsqueeze(0).to(device)
    h0 = batch['h0'].unsqueeze(0).to(device)
    trace = json.loads((a.run / 'TRAINING_TRACE.jsonl').open().readline())
    if trace['step'] != 1 or trace['batch_sha256'] != batch_hash:
        raise ValueError('trace and augmented batch differ')

    def forward():
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            source = runner._source_logits(model, image)
            states, _ = runner.legacy.iterative_refine_o(
                image, source, refiner, 2, torch.zeros_like(image), indices=None
            )
            final = states[-1]
            q_source = F.softplus(-source.float())[h0].mean()
            q_final = F.softplus(-final.float())[h0].mean()
        return source, final, q_source, q_final

    before_source, before_final, before_q_source, before_q_final = forward()
    if abs(float(before_q_final) - trace['Q_H']['mean_softplus_minus_final']) > 1e-5:
        raise ValueError('pre-step final Q_H differs from actual training trace')
    controls, meta = response.make_matched_controls(
        batch['mask'].bool().numpy(), batch['valid'][0].numpy(),
        batch['h0'].numpy(), before_source[0].float().cpu().numpy(),
    )
    matched = controls[controls[:, 3] >= 0]
    if len(matched) == 0:
        raise ValueError('no matched negative controls')
    member = torch.as_tensor(matched[:, 0], device=device, dtype=torch.long)
    yy = torch.as_tensor(matched[:, 3], device=device, dtype=torch.long)
    xx = torch.as_tensor(matched[:, 4], device=device, dtype=torch.long)
    neg_before = F.softplus(before_final[0, member, yy, xx].float()).mean()
    checkpoint = a.run / 'checkpoints' / 'step_00001.pth'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model'], strict=True)
    refiner.load_state_dict(payload['refiner'], strict=True)
    after_source, after_final, after_q_source, after_q_final = forward()
    neg_after = F.softplus(after_final[0, member, yy, xx].float()).mean()
    output = {
        'status': 'COMPLETE', 'test_used': False, 'optimizer_updates': 0,
        'batch_sha256': batch_hash, 'H0_members': int(h0.sum()),
        'C0_sha256': runner.sha256_file(Path(lock['checkpoint'])),
        'step1_checkpoint_sha256': runner.sha256_file(checkpoint),
        'Q_final_before': float(before_q_final), 'Q_final_after': float(after_q_final),
        'Q_final_actual_delta': float(after_q_final - before_q_final),
        'Q_source_before': float(before_q_source), 'Q_source_after': float(after_q_source),
        'Q_source_actual_delta': float(after_q_source - before_q_source),
        'Q_final_gradient_dot_actual_delta': sum(trace['Q_H']['gradient_dot_actual_delta'].values()),
        'matched_negative_controls': len(matched),
        'matched_negative_pair_sha256': meta['pair_sha256'],
        'matched_negative_final_BCE_before': float(neg_before),
        'matched_negative_final_BCE_after': float(neg_after),
        'matched_negative_final_BCE_actual_delta': float(neg_after - neg_before),
        'H0_final_recovered_before': int((torch.sigmoid(before_final.float())[h0] >= .5).sum()),
        'H0_final_recovered_after': int((torch.sigmoid(after_final.float())[h0] >= .5).sum()),
    }
    a.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
