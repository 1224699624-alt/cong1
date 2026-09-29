"""Round2 bounded real-batch acceptance. No epochs, pilot, or test access.

Only one train batch per dataset can update an in-memory model copy once.
Validation batches are forward-only. Native files are opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from local_supervision_v3 import local_supervision
from r349_parameter_gradient_gate_v3 import route_parameter_gradients_v3, assign_gradients_v3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def high(output):
    return output[0] if isinstance(output, (list, tuple)) else output


def batch_stats(value):
    value = value.detach().float()
    return {'shape': list(value.shape), 'min': float(value.min()), 'max': float(value.max()),
            'mean': float(value.mean()), 'nonzero': int(torch.count_nonzero(value))}


def tsrs_assets(root):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
    base = root / 'outputs/nnunet/r317_image_centers_only'
    prepared = base / 'data/nnUNet_preprocessed/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D'
    checkpoint = base / 'r202_two_channel_zero_prior.pth'
    assert sha(checkpoint) == '55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b'
    plans = PlansManager(str(prepared / 'nnUNetPlans.json'))
    config = plans.get_configuration('2d')
    model = nnUNetTrainer.build_network_architecture(plans, config, 2, 2, True)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['network_weights'], strict=True)
    dataset = nnUNetDatasetBlosc2(str(prepared / 'nnUNetPlans_2d'))
    split_file = prepared / 'splits_final.json'
    splits = json.loads(split_file.read_text())[0]
    batches = []
    for split in ('train', 'val'):
        case = sorted(splits[split])[0]
        data, seg, _, props = dataset.load_case(case)
        data = np.asarray(data[:])[:, 0].copy()
        seg = np.asarray(seg[:])[:, 0].copy()
        if data.shape[0] != 2 or data.shape[-2:] != seg.shape[-2:]:
            raise ValueError('TSRS preprocessed channel/coordinate mismatch')
        # Exact native pixel crop, without resizing. Training uses the native
        # plan patch size; val is a whole preprocessed case padded for strides.
        h, w = data.shape[-2:]
        if split == 'train':
            h, w = min(h, config.patch_size[0]), min(w, config.patch_size[1])
        data = torch.from_numpy(data[:, :h, :w]).unsqueeze(0)
        target = torch.from_numpy(seg[:, :h, :w]).unsqueeze(0).float()
        pad = (0, (-w) % 256, 0, (-h) % 256)
        valid = F.pad((target >= 0), pad, value=False)
        data, target = F.pad(data, pad), F.pad(target.clamp_min(0), pad)
        batches.append({'case': case, 'split': split, 'image': data[:, :1], 'seam': data[:, 1:2],
                        'target': target, 'valid': valid,
                        'coordinate_record': {'loader': 'nnUNetDatasetBlosc2', 'original_shape': list(seg.shape),
                                              'crop_origin': [0, 0], 'crop_hw': [h, w], 'padding_lrtb': list(pad),
                                              'spacing': list(props['spacing']), 'resize': False,
                                              'normalization': 'existing R317 preprocessed image and prior ZScoreNormalization'},
                        'input_artifact_sha256': sha(prepared/'nnUNetPlans_2d'/f'{case}.b2nd')})
    native_base = DC_and_CE_loss({'batch_dice': config.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
                               {}, weight_ce=1, weight_dice=1, ignore_label=None,
                               dice_class=MemoryEfficientSoftDiceLoss)
    def base_loss(output, target):
        values = output if isinstance(output, (list, tuple)) else [output]
        weights = np.array([1 / 2**i for i in range(len(values))]); weights[-1] = 0
        weights /= weights.sum()
        labels = [F.interpolate(target, z.shape[-2:], mode='nearest') for z in values]
        return DeepSupervisionWrapper(native_base, weights)(values, labels)
    return model, model, batches, checkpoint, base_loss, {'model_input_channels': 2, 'strict_load': True,
                                                         'split_sha256': sha(split_file), 'deep_supervision': True}


def ram_assets(root):
    from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, dice_bce_logits
    from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
    from train_r322_ram_nnunet_r317_seam_prior import RamNnUNetTwoChannel, initialize_from_r285
    checkpoint = root / 'outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth'
    assert sha(checkpoint) == '88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9'
    native = NnUNetMultiLabelPrior()
    native.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model = RamNnUNetTwoChannel()
    conversion = initialize_from_r285(model, checkpoint, torch.device('cpu'))
    batches = []
    for split in ('train', 'val'):
        ds = NativeWristDataset(root/'data/remote_variants/RAM-W600', split, augment=False)
        # Right wrist deliberately exercises the canonical flip contract.
        index = next(i for i, path in enumerate(ds.mask_files) if path.stem.endswith('_R'))
        b = ds[index]; image = b['image'][None]; target = b['mask'][None]; valid = b['valid'][None]
        path = root/'outputs/priors/r323_ram_native_r317_single_seed'/split/(b['case']+'.npy')
        prior = np.load(path)
        if prior.shape != (384, 384) or not np.isfinite(prior).all():
            raise ValueError('R323 map must have documented canonical 384x384 coordinates')
        h, w = b['original_hw'].tolist()
        # Explicit historical inverse resize, not an evaluator repair.
        seam = F.interpolate(torch.from_numpy(prior.copy())[None, None], (h, w), mode='bilinear', align_corners=False)
        seam = F.pad(seam, (0, image.shape[-1]-w, 0, image.shape[-2]-h))
        batches.append({'case': b['case'], 'split': split, 'image': image, 'target': target, 'valid': valid,
                        'seam': seam, 'input_artifact_sha256': sha(path),
                        'coordinate_record': {'loader': 'NativeWristDataset', 'original_hw': [h, w],
                                              'image_gt_flip_count': 1, 'prior_additional_flip_count': 0,
                                              'prior_source_hw': [384, 384], 'prior_transform': 'bilinear to native then right/bottom pad',
                                              'spacing': None, 'distance_units': 'pixels', 'image_resize': False}})
    return native, model, batches, checkpoint, dice_bce_logits, conversion


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', choices=['TSRS', 'RAM'], required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f'N0_EQUIVALENCE_{args.dataset}.json'
    if out.exists():
        raise FileExistsError(out)
    from structural_prior_v3 import build_backbone_input
    torch.manual_seed(3492); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    native, model, batches, checkpoint, base_loss, conversion = (tsrs_assets if args.dataset == 'TSRS' else ram_assets)(args.root)
    original_sha = sha(checkpoint)
    native = native.to(device).eval(); model = model.to(device).eval()
    forward_native = (lambda image: native(torch.cat([image, torch.zeros_like(image)], 1))) if args.dataset == 'TSRS' else (lambda image: native(image, False)[0])
    reports = []
    for batch in batches:
        image, seam, target, valid = [batch[k].to(device) for k in ('image', 'seam', 'target', 'valid')]
        entry_records = {}
        def capture(module, inputs):
            x = inputs[0]
            entry_records['input'] = {'channel0': batch_stats(x[:, :1]), 'channel1': batch_stats(x[:, 1:2]),
                                      'original_image_unchanged': bool(torch.equal(x[:, :1], image)),
                                      'seam_channel_equals_map': bool(torch.equal(x[:, 1:2], seam))}
        handle = model.register_forward_pre_hook(capture)
        with torch.no_grad():
            n0 = high(forward_native(image))
            off = high(model(build_backbone_input(image, seam, 'A')))
            delta = float((n0-off).abs().max())
            mask = (lambda z: z.argmax(1)) if args.dataset == 'TSRS' else (lambda z: z.sigmoid() >= .5)
            mask_diff = int(torch.count_nonzero(mask(n0) != mask(off)))
            on = high(model(build_backbone_input(image, seam, 'B')))
            entry_b = dict(entry_records['input'])
            identity_b = float((on-n0).abs().max())
            teacher = (n0[:, 1:2]-n0[:, :1]).sigmoid() if args.dataset == 'TSRS' else n0.sigmoid()
            margin_error = float(((n0[:, 1:2]-n0[:, :1]).sigmoid()-n0.softmax(1)[:, 1:2]).abs().max()) if args.dataset == 'TSRS' else None
        handle.remove()
        row = {k:v for k,v in batch.items() if k not in ('image','seam','target','valid')}
        row.update({'status': 'PASS' if delta <= 1e-5 and mask_diff == 0 and identity_b <= 1e-5 else 'FAIL',
                    'n0_max_abs_logit_difference': delta, 'n0_mask_mismatch_pixels': mask_diff,
                    'B_zero_weight_identity_max_abs': identity_b, 'B_backbone_entry': entry_b,
                    'margin_softmax_max_abs': margin_error, 'tensor_hw': list(image.shape[-2:]), 'tolerance': 1e-5})
        if row['status'] != 'PASS':
            raise RuntimeError(row)
        reports.append(row)
    # One real train batch, native base loss with its original deep supervision.
    b = batches[0]
    image, seam, target, valid = [b[k].to(device) for k in ('image','seam','target','valid')]
    model.train(); model.zero_grad(set_to_none=True)
    with torch.no_grad():
        native.eval(); ref = high(forward_native(image))
        teacher = (ref[:, 1:2]-ref[:, :1]).sigmoid() if args.dataset == 'TSRS' else ref.sigmoid()
    model.train()
    # Same bf16 policy for this train-only mechanical check; N0 comparison above is FP32.
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(build_backbone_input(image, seam, 'B'))
        loss = base_loss(prediction, target)
    z = high(prediction).float()
    if args.dataset == 'TSRS':
        z = z[:, 1:2]-z[:, :1]; known = torch.zeros_like(target, dtype=torch.bool)
    else:
        known = torch.ones_like(target, dtype=torch.bool)
    terms, states = local_supervision(z, target, known, target.any(1,keepdim=True), valid, valid,
                                      seam, teacher, enable_seam=True, enable_overlap=False)
    named = list(model.named_parameters()); params = [v for _,v in named]
    gradients, routing = route_parameter_gradients_v3(params, loss,
        [('separation',terms['separation'],.035),('preserve',terms['preserve'],.10)])
    assign_gradients_v3(params, gradients)
    finite = all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
    first = next(p for p in params if p.ndim == 4 and p.shape[1] == 2)
    seam_gradient_norm = float(first.grad[:, 1].float().norm())
    if not finite or seam_gradient_norm == 0:
        raise RuntimeError('invalid real seam channel gradient')
    torch.nn.utils.clip_grad_norm_(params, 5.)
    optimizer = torch.optim.AdamW(params, lr=2e-6, weight_decay=1e-4)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        after = high(model(build_backbone_input(image,seam,'B')))
        # Demonstrate actual learned seam-channel reachability after the one
        # allowed temporary update; this is not a performance comparison.
        changed = high(model(build_backbone_input(image,torch.zeros_like(seam),'B')))
        seam_effect = float((after-changed).abs().max())
    with tempfile.TemporaryDirectory(prefix='round2_reload_',dir=args.output) as tmp:
        path = Path(tmp)/'temporary_model.pth'; torch.save(model.state_dict(),path)
        state = torch.load(path,map_location=device,weights_only=False)
        model.load_state_dict(state,strict=True)
        with torch.no_grad():
            reloaded = high(model(build_backbone_input(image,seam,'B')))
        reload_delta = float((after-reloaded).abs().max())
    final_sha = sha(checkpoint)
    report = {'status': 'PASS' if finite and reload_delta == 0 and original_sha == final_sha else 'FAIL',
              'dataset': args.dataset, 'execution_environment': 'authorized remote GPU; not local full-validation reproduction',
              'checkpoint': str(checkpoint), 'source_sha256':original_sha, 'source_file_unchanged': original_sha == final_sha,
              'conversion': conversion, 'native_equivalence': reports,
              'actual_backbone_input_contract': 'original normalized image plus independent aligned seam heatmap; A/C seam zero',
              'train_one_step': {'case':b['case'],'finite_gradients':finite,'seam_channel_gradient_norm':seam_gradient_norm,
                                 'seam_effect_after_one_step':seam_effect,'strict_reload_max_abs':reload_delta,
                                 'params':len(params),'unused_gradient_count':sum(g is None for g in gradients),
                                 'routing':[asdict(r) if is_dataclass(r) else r for r in routing],
                                 'observation':states['overlap_supervision'], 'state_mass':states['masses']},
              'arm_status': {'A':'PASS real N0 forward','B':'PASS real forward/backward/temporary step',
                             'C':'BLOCKED image-only trained overlap weights not verified',
                             'D':'BLOCKED image-only trained overlap weights not verified'},
              'test_read':False, 'epochs_run':0,'pilot_run':False,'full_val_reproduction_this_round':'NOT_RUN',
              'source_version':{p:sha(Path(__file__).parent/p) for p in
                                ['check_round2_real_batch.py','structural_prior_v3.py','local_supervision_v3.py','r349_parameter_gradient_gate_v3.py']}}
    out.write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
