"""Read-only RAM C0 O ingress trace at backbone/refiner native tensors."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v31_ram_mature as legacy
from r332_v31_port import features_with_o


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    lock = json.loads(args.lock.read_text())
    for raw, expected in lock['file_hashes'].items():
        if legacy.sha(raw) != expected:
            raise ValueError(f'SHA mismatch: {raw}')
    torch.set_num_threads(64)
    device = torch.device('cuda')
    _, model, refiner, generator, _, _ = legacy.load_components(
        SimpleNamespace(r332=Path(lock['r332']), overlap=Path(lock['overlap']),
                        overlap_sha=lock['overlap_sha256']), device
    )
    state = torch.load(lock['fresh100'], map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    refiner.load_state_dict(state['refiner'], strict=True)
    for module in (model, refiner, generator):
        module.eval().requires_grad_(False)
    o_weight = refiner.enc1.net[0].weight[:, 6].detach()
    encoder_weight = model.backbone.encoder.stages[0][0].convs[0].conv.weight.detach()
    dataset = legacy.NativeWristDataset(Path(lock['dataset']), 'val', False)
    if len(dataset) != 69:
        raise ValueError('not RAM val69')
    cases = []
    with torch.inference_mode():
        for index in range(3):
            sample = dataset[index]
            image = sample['image'].unsqueeze(0).to(device)
            zero = torch.zeros_like(image)
            source = model(torch.cat((image, zero), dim=1))
            o = generator(image).sigmoid()
            probability = source.sigmoid()
            member = torch.arange(3, device=device)
            feature_zero = features_with_o(image, probability, member,
                                           probability[:, member], zero)
            feature_real = features_with_o(image, probability, member,
                                           probability[:, member], o)
            conv = refiner.enc1.net[0]
            conv_zero = conv(feature_zero)
            conv_real = conv(feature_real)
            states_zero, _ = legacy.iterative_refine_o(image, source, refiner, 2, zero)
            states_real, _ = legacy.iterative_refine_o(image, source, refiner, 2, o)
            valid = sample['valid'].unsqueeze(0).to(device)
            cases.append({
                'case': sample['case'],
                'image_shape': list(image.shape),
                'backbone_input_shape': [1, 2, *image.shape[-2:]],
                'backbone_second_input_abs_max': float(zero.abs().max()),
                'O_shape': list(o.shape),
                'O_valid_min_mean_max': [float(o[valid].min()), float(o[valid].mean()),
                                         float(o[valid].max())],
                'refiner_feature_shape': list(feature_real.shape),
                'refiner_O_input_abs_max': float(feature_real[:, 6].abs().max()),
                'refiner_feature_O_minus_zero_abs_max': float((feature_real - feature_zero).abs().max()),
                'enc1_conv_output_O_minus_zero_abs_max': float((conv_real - conv_zero).abs().max()),
                'step1_logit_O_minus_zero_abs_max': float((states_real[1] - states_zero[1]).abs().max()),
                'step2_logit_O_minus_zero_abs_max': float((states_real[2] - states_zero[2]).abs().max()),
                'step2_mask_changed_members': int(((states_real[2].sigmoid() >= .5) !=
                                                   (states_zero[2].sigmoid() >= .5)).sum()),
            })
    result = {
        'status': 'PASS', 'optimizer_updates': 0, 'test_used': False,
        'C0_sha256': legacy.sha(lock['fresh100']),
        'O_sha256': legacy.sha(lock['overlap']),
        'backbone_first_conv_second_input_weight_nonzero': int(torch.count_nonzero(encoder_weight[:, 1])),
        'refiner_enc1_O_weight_nonzero': int(torch.count_nonzero(o_weight)),
        'cases': cases,
        'interpretation': 'Real O reaches the appended refiner feature but is annihilated by zero seventh-channel enc1 weight; source backbone receives image+zero. No evidence of later downsampling loss can be inferred before this ingress is opened.',
    }
    if not all(case['refiner_O_input_abs_max'] > 0 and
               case['enc1_conv_output_O_minus_zero_abs_max'] == 0 and
               case['step2_logit_O_minus_zero_abs_max'] == 0 for case in cases):
        result['status'] = 'FAIL'
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'cases': cases}, indent=2))


if __name__ == '__main__':
    main()
