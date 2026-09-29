"""Zero-update paired CUDA profile of O=zero versus frozen real O source-head adapter."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import run_ram_o_source_head_pair as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text())
    runner._verify_lock_hashes(lock, lock_path)
    steps, _, _ = runner._load_schedule(lock, lock_path)
    batch, batch_sha = runner._load_step_payload(steps[0])
    device = torch.device('cuda')
    image = batch['image'].unsqueeze(0).to(device)
    target = batch['mask'].float().unsqueeze(0).to(device)
    valid = batch['valid'].bool().unsqueeze(0).to(device)
    h0 = batch['h0'].bool().unsqueeze(0).to(device)
    indices = torch.tensor(batch['indices'], dtype=torch.long, device=device)
    runner.seed_all(int(lock['seed']))
    rows = []
    first_source = None
    for arm in runner.ARMS:
        _, model, refiner, generator, optimizer, scaler, named_groups, _, _, _ = \
            runner._prepare_components(lock, lock_path, arm, device)
        attachment = runner._prepare_o_adapter(
            model, generator, optimizer, named_groups, use_real_o=arm == 'OHEAD_REAL'
        )
        runner._configure_modes(model, refiner)
        attachment.adapter.train()
        original_selector = runner.legacy.choose_pair_instances

        def selected(_probability, _target, count):
            if count != 4:
                raise ValueError('native selector must request four')
            return indices

        runner.legacy.choose_pair_instances = selected
        try:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            with torch.cuda.amp.autocast(enabled=True):
                captured = []
                hook = model.register_forward_hook(lambda _m, _i, output: captured.append(output))
                try:
                    native, _, _, _, chosen, _ = runner.legacy.loss_for_batch(
                        model, refiner, generator, image, target, valid, False
                    )
                finally:
                    hook.remove()
                if len(captured) != 1 or chosen.tolist() != batch['indices']:
                    raise ValueError('native source/selection differs')
                source = captured[0]
                extra = runner.EXTRA_LAMBDA * F.softplus(-source.float())[h0].sum() / batch['denominator_c0']
                total = native + extra
            if not all(bool(torch.isfinite(x)) for x in (native, extra, total)):
                raise FloatingPointError('nonfinite forward')
            source_cpu = source.detach().float().cpu()
            if first_source is None:
                first_source = source_cpu
                source_max_abs_diff = 0.0
            else:
                source_max_abs_diff = float((source_cpu - first_source).abs().max())
            scaler.scale(total).backward()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            scale = float(scaler.get_scale())
            gradients = {}
            for group, named in named_groups.items():
                values = [parameter.grad.float() / scale for _, parameter in named if parameter.grad is not None]
                gradients[group] = {
                    'present': len(values),
                    'l2': math.sqrt(sum(float(value.square().sum()) for value in values)),
                    'finite': all(bool(torch.isfinite(value).all()) for value in values),
                }
            generator_grad_present = sum(parameter.grad is not None for parameter in generator.parameters())
            rows.append({
                'arm': arm, 'source_max_abs_diff_from_zero_arm_at_C0': source_max_abs_diff,
                'native_loss': float(native.detach()), 'extra_source_H0_loss': float(extra.detach()),
                'total_loss': float(total.detach()), 'selected_four': chosen.tolist(),
                'H0_members': int(h0.sum()), 'elapsed_forward_backward_s': elapsed,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated(device),
                'adapter_weight_nonzero_initial': int(torch.count_nonzero(attachment.adapter.weight.detach())),
                'generator_grad_present': generator_grad_present,
                'gradients': gradients, 'AMP_initial_scale': scale,
            })
        finally:
            runner.legacy.choose_pair_instances = original_selector
            attachment.close()
            del model, refiner, generator, optimizer, scaler, attachment
            torch.cuda.empty_cache()
    if any(abs(row['native_loss'] - 0.023890312761068344) > 1e-5 or
           abs(row['extra_source_H0_loss'] - 0.009006000123918056) > 1e-5 or
           row['source_max_abs_diff_from_zero_arm_at_C0'] > 1e-6 or
           row['generator_grad_present'] != 0 or
           not all(value['finite'] for value in row['gradients'].values()) for row in rows):
        raise ValueError('paired initial parity or finite gradient check failed')
    if rows[0]['gradients']['o_source_adapter']['l2'] != 0 or \
            rows[1]['gradients']['o_source_adapter']['l2'] <= 0:
        raise ValueError('adapter gradients do not isolate zero versus real O')
    output = {'status': 'PASS', 'optimizer_updates': 0, 'test_used': False,
              'batch_sha256': batch_sha, 'C0_sha256': runner.sha256_file(Path(lock['checkpoint'])),
              'rows': rows}
    args.out.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
