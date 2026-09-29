"""Isolated R332/V3.1 diagnostics. Never resumes or overwrites historical runs."""
import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import run_v31_ram_mature as legacy


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def stats(t):
    x = t.detach().float().reshape(-1)
    good = x[torch.isfinite(x)]
    if not good.numel():
        return {'shape': list(t.shape), 'finite_fraction': 0.}
    # Fixed strided sample; no RNG consumption.
    sample = good[::max(1, good.numel() // 4096)]
    return {'shape': list(t.shape), 'mean': float(good.mean()), 'std': float(good.std(unbiased=False)),
            'min': float(good.min()), 'max': float(good.max()), 'norm': float(good.norm()),
            'zero_fraction': float((good == 0).float().mean()),
            'finite_fraction': float(torch.isfinite(x).float().mean()),
            'q05_q50_q95': torch.quantile(sample, torch.tensor([.05, .5, .95], device=x.device)).tolist()}


def rng_get():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def rng_set(s):
    random.setstate(s[0]); np.random.set_state(s[1]); torch.set_rng_state(s[2]); torch.cuda.set_rng_state_all(s[3])


def torch_rng_unchanged(before, after):
    return torch.equal(before[2], after[2]) and len(before[3]) == len(after[3]) and all(
        torch.equal(b, a) for b, a in zip(before[3], after[3]))


def grouping(name):
    if name.startswith('refiner.'):
        return '.'.join(name.split('.')[:2])
    fields = name.split('.')
    return '.'.join(fields[:5])


def grad_summary(named, grads):
    out = {}
    for (name, p), g in zip(named, grads):
        key = grouping(name)
        r = out.setdefault(key, {'squared_norm': 0., 'elements': 0, 'none_elements': 0,
                                 'zero_elements': 0, 'nonfinite_elements': 0})
        r['elements'] += p.numel()
        if g is None:
            r['none_elements'] += p.numel(); continue
        g = g.detach().float()
        r['squared_norm'] += float(g.square().sum())
        r['zero_elements'] += int((g == 0).sum())
        r['nonfinite_elements'] += int((~torch.isfinite(g)).sum())
    for r in out.values():
        r['norm'] = r.pop('squared_norm') ** .5
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lock', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    lock = json.loads(a.lock.read_text())
    if lock.get('cpu_threads') is not None:
        cpu_threads = int(lock['cpu_threads'])
        if cpu_threads < 1:
            raise ValueError('cpu_threads must be positive')
        torch.set_num_threads(cpu_threads)
    matched_reference = None
    if lock.get('matched_reference_trace'):
        matched_reference = [json.loads(line) for line in Path(lock['matched_reference_trace']).read_text().splitlines()]
        if len(matched_reference) != lock['updates']:
            raise ValueError('Reference length does not match update budget')
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    for path, expected in lock['file_hashes'].items():
        if legacy.sha(path) != expected:
            raise ValueError('hash mismatch: ' + path)
    start = time.monotonic()
    legacy.seed_all(lock['seed'])
    device = torch.device('cuda')
    args = SimpleNamespace(r332=Path(lock['r332']), overlap=Path(lock['overlap']),
                           overlap_sha=lock['overlap_sha'])
    payload, model, refiner, generator, bmap, rmap = legacy.load_components(args, device)
    if lock.get('continuation_checkpoint'):
        payload = torch.load(lock['continuation_checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model'], strict=True)
        refiner.load_state_dict(payload['refiner'], strict=True)
    negative_guard_enabled = bool(lock.get('negative_guard_enabled', False))
    negative_guard_band = float(lock.get('negative_guard_band', .25)) if negative_guard_enabled else .25
    if negative_guard_enabled and negative_guard_band <= 0:
        raise ValueError('negative_guard_band must be positive')
    teacher_model = teacher_refiner = None
    if negative_guard_enabled:
        # Snapshot the loaded continuation before trainability changes or hooks.
        teacher_model = copy.deepcopy(model).eval().requires_grad_(False)
        teacher_refiner = copy.deepcopy(refiner).eval().requires_grad_(False)
    legacy.set_joint_trainability(model, True)
    model.eval(); refiner.train()
    for mod in [model.backbone.decoder.stages[3], model.backbone.decoder.stages[4],
                model.backbone.decoder.transpconvs[3], model.backbone.decoder.transpconvs[4],
                model.backbone.decoder.seg_layers[4]]:
        mod.train()
    named = [('model.' + n, t) for n, t in model.named_parameters() if t.requires_grad]
    named += [('refiner.' + n, t) for n, t in refiner.named_parameters() if t.requires_grad]
    params = [t for _, t in named]
    decoder = [t for t in model.parameters() if t.requires_grad]
    optimizer = torch.optim.AdamW([{'params': list(refiner.parameters()), 'lr': lock.get('refiner_lr_override', 8e-5)},
                                   {'params': decoder, 'lr': 7e-6}], weight_decay=1e-4)
    restored_adam = None
    if lock.get('restore_adam_from'):
        from r332_adam_state_port import restore_r332_adam
        restored_adam = restore_r332_adam(optimizer, model, refiner, lock['restore_adam_from'])
        write(a.out / 'ADAM_STATE_MAPPING.json', restored_adam)
    scaler = torch.cuda.amp.GradScaler()
    if lock.get('continuation_checkpoint'):
        if restored_adam is not None: raise ValueError('Choose one optimizer source')
        optimizer.load_state_dict(copy.deepcopy(payload['optimizer']))
        scaler.load_state_dict(copy.deepcopy(payload['scaler']))
        assert [g['lr'] for g in optimizer.param_groups] == [lock['refiner_lr_override'],7e-6]
    registered = [id(t) for g in optimizer.param_groups for t in g['params']]
    assert len(registered) == len(set(registered)) and set(registered) == set(map(id, params))
    teacher_frozen = not negative_guard_enabled or all(
        not p.requires_grad for root in (teacher_model, teacher_refiner) for p in root.parameters())
    teacher_excluded = not negative_guard_enabled or all(
        id(p) not in set(registered) for root in (teacher_model, teacher_refiner) for p in root.parameters())
    initial = {n: t.detach().cpu().clone() for n, t in named}
    audit = {'history_mode': ('PAIRED_CURRENT_CHECKPOINT_CONTINUATION' if lock.get('continuation_checkpoint') else 'CURRENT_STATE_NEW_DIAGNOSIS_RESTORED_ADAM_ONLY' if restored_adam else 'CURRENT_STATE_NEW_DIAGNOSIS_FRESH_OPTIMIZER'),
             'restored_adam': restored_adam,
             'historical_checkpoint_keys': list(payload), 'historical_optimizer_available': 'optimizer' in payload,
             'optimizer_coverage_exact': True, 'backbone_mapping': bmap, 'refiner_mapping': rmap,
             'parameters': [{'name': n, 'shape': list(t.shape), 'group': 0 if n.startswith('refiner.') else 1,
                             'initial_norm': float(t.norm())} for n, t in named],
             'frozen_generator_expected': True, 'scheduler': lock.get('scheduler', 'constant LR during <100-step epoch prefix; no scheduler step due'),
             'routing': 'none', 'state_decay': 'none', 'deep_supervision': 'two refiner steps 1/3,2/3; source native BCE+Dice .1',
             'train_eval': {n: m.training for n, m in model.named_modules() if n.endswith(('stages.3', 'stages.4', 'encoder'))},
             'cpu_threads': torch.get_num_threads(),
             'negative_guard': {'enabled': negative_guard_enabled, 'band': negative_guard_band,
                                'teacher_frozen': teacher_frozen,
                                'teacher_excluded_from_optimizer': teacher_excluded}}
    write(a.out / 'PARAMETER_AUDIT.json', audit)
    data = Path(lock['dataset'])
    train = legacy.NativeWristDataset(data, 'train', True)
    loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(lock['seed']))
    source_aux_box = {}
    if lock.get('source_auxiliary_mode') or negative_guard_enabled:
        from r332_hard_source_supervision import hard_source_loss, local_negative_guard_loss
        teacher_rng_checked = [False]
        teacher_rng_unchanged = [None]
        def capture_source(module, inputs, output): source_aux_box['logits'] = output
        model.register_forward_hook(capture_source)
        original_batch_loss = legacy.loss_for_batch
        def extended_batch_loss(model_arg,refiner_arg,generator_arg,image,target,valid,use_o):
            native,parts,stats,o,indices,supervision = original_batch_loss(model_arg,refiner_arg,generator_arg,image,target,valid,use_o)
            extra,meta = hard_source_loss(source_aux_box['logits'],target,valid,indices)
            source_aux_box['meta'] = meta
            parts = {**parts,'original_total':native,'unselected_hard_source':extra}
            total = native + extra if lock.get('source_aux_enabled', False) else native
            if negative_guard_enabled:
                image_fp32 = image.float()
                zero_o = torch.zeros_like(image_fp32)
                teacher_rng_before = rng_get() if not teacher_rng_checked[0] else None
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    teacher_base = teacher_model(torch.cat((image_fp32, zero_o), 1))
                    teacher_states, _ = legacy.iterative_refine_o(
                        image_fp32, teacher_base, teacher_refiner, 2, zero_o)
                    teacher_final = teacher_states[-1]
                if teacher_rng_before is not None:
                    teacher_rng_unchanged[0] = torch_rng_unchanged(teacher_rng_before, rng_get())
                    if not teacher_rng_unchanged[0]:
                        raise RuntimeError('frozen teacher consumed torch RNG state')
                    teacher_rng_checked[0] = True
                guard, guard_meta = local_negative_guard_loss(
                    source_aux_box['logits'], target, valid, teacher_final,
                    meta['denominator'], band=negative_guard_band)
                guard_meta = {**guard_meta, 'teacher_rng_unchanged': bool(teacher_rng_unchanged[0])}
                source_aux_box['negative_guard_meta'] = guard_meta
                parts['negative_guard'] = guard
                total = total + guard
            return total,parts,stats,o,indices,supervision
        legacy.loss_for_batch = extended_batch_loss
    activations = {}; active = [False]; handles = []
    modules = {'encoder0': model.backbone.encoder.stages[0], 'bottleneck': model.backbone.encoder.stages[-1],
               'decoder3': model.backbone.decoder.stages[3], 'decoder4': model.backbone.decoder.stages[4],
               'refiner_enc1': refiner.enc1, 'refiner_mid': refiner.mid,
               'refiner_gate': refiner.gate, 'refiner_residual': refiner.residual}
    for key, module in modules.items():
        def hook(m, inp, out, key=key):
            if active[0]:
                row = {'input': stats(inp[0]), 'output': stats(out)}
                if key == 'refiner_enc1':
                    row['channels'] = [stats(inp[0][:, i]) for i in range(inp[0].shape[1])]
                if key == 'refiner_gate': row['sigmoid'] = stats(out.sigmoid())
                if key == 'refiner_residual': row['tanh'] = stats(out.tanh())
                activations.setdefault(key, []).append(row)
        handles.append(module.register_forward_hook(hook))
    original_completion = legacy.completion_loss
    original_surface = legacy.soft_surface_nsd_loss
    original_volume = legacy.volume_losses
    terms = {}; extra = {}; counts = [0]
    def completion(z, target, local, delta, weight):
        loss, values = original_completion(z, target, local, delta, weight)
        k = counts[0]; counts[0] += 1
        terms[f'step{k+1}_completion'] = (loss, (k + 1) / 3)
        if active[0]:
            prob = z.float().sigmoid(); w = 1 + 3 * local
            pixel_bce = F.binary_cross_entropy_with_logits(z.float(), target.float(), reduction='none')
            v = extra['valid'].expand_as(target)
            pad_bce = (pixel_bce * w * (~v)).sum() / w.sum()
            extra[f'step{k+1}'] = {'local_fraction': float(local.mean()), 'bce_denominator': float(w.sum()),
                'padding_bce_share': float(pad_bce / ((pixel_bce*w).sum()/w.sum()).clamp_min(1e-20)),
                'padding_fraction': float((~v).float().mean()), 'residual': stats(delta),
                'positive_pixels': int(target.sum()), 'valid_pixels': int(v.sum())}
            terms[f'step{k+1}_padding_bce_diagnostic_only'] = (pad_bce, 0.)
            multi = target[:, 0].sum(0) >= 2
            if bool(multi.any()):
                y, x = torch.nonzero(multi)[0].tolist()
                extra[f'step{k+1}']['true_multimember_pixel'] = {'yx': [y,x],
                    'targets': target[:,0,y,x].tolist(), 'probabilities': prob[:,0,y,x].tolist(),
                    'unweighted_bce_logit_gradient_sign': torch.sign(prob[:,0,y,x]-target[:,0,y,x]).tolist()}
        return loss, values
    def surface(prob, target):
        loss = original_surface(prob, target); terms[f'step{counts[0]}_surface'] = (loss, .03*counts[0]/3); return loss
    def volume(prob, target):
        iv, pv = original_volume(prob, target)
        terms[f'step{counts[0]}_instance_volume'] = (iv, .005*counts[0]/3)
        terms[f'step{counts[0]}_pair_volume'] = (pv, .01*counts[0]/3)
        return iv, pv
    legacy.completion_loss = completion; legacy.soft_surface_nsd_loss = surface; legacy.volume_losses = volume
    successful = 0; cases = set(); timings = []
    for step, batch in enumerate(loader, 1):
        if step > lock['updates']: break
        if time.monotonic() - start > lock['max_seconds'] or shutil.disk_usage(a.out).free < 4*1024**3:
            raise RuntimeError('time/disk budget')
        tick = time.monotonic(); active[0] = step in lock['detailed_steps']; activations.clear(); terms.clear(); counts[0] = 0
        image, target, valid = [batch[k].to(device) for k in ('image','mask','valid')]
        extra = {'valid': valid.bool()}; optimizer.zero_grad(set_to_none=True)
        random_state = rng_get()
        with torch.cuda.amp.autocast():
            loss, parts, details, o, indices, supervision = legacy.loss_for_batch(model,refiner,generator,image,target,valid,lock['use_o'])
        after_random = rng_get()
        for key, weight in [('global', .1), ('monotonic', .05), ('contraction', .01)]: terms[key] = (parts[key],weight)
        if 'unselected_hard_source' in parts:
            terms['unselected_hard_source']=(parts['unselected_hard_source'],1. if lock.get('source_aux_enabled', False) else 0.)
        if 'negative_guard' in parts:
            terms['negative_guard']=(parts['negative_guard'],1.)
        row = {'step':step, 'case':batch['case'][0], 'mode':audit['history_mode'],
               'batch_sha256':hashlib.sha256(b''.join(batch[k].numpy().tobytes() for k in ('image','mask','valid'))).hexdigest(),
               'loss':float(loss.detach()), 'loss_terms':{k:{'raw':float(v.detach()),'coefficient':w,'weighted':float(v.detach())*w} for k,(v,w) in terms.items()},
               'lr':[g['lr'] for g in optimizer.param_groups], 'supervision':supervision,
               'prior':stats(o), 'activations':dict(activations)}
        row['original_loss_before_source_aux']=float(parts.get('original_total',loss).detach())
        if 'meta' in source_aux_box: row['source_auxiliary']=dict(source_aux_box['meta'])
        if 'negative_guard_meta' in source_aux_box: row['negative_guard']=dict(source_aux_box['negative_guard_meta'])
        if matched_reference is not None:
            reference_row = matched_reference[step-1]
            if row['batch_sha256'] != reference_row['batch_sha256']:
                raise ValueError('Matched data sequence failed before optimizer step')
            if step == 1 and row['original_loss_before_source_aux'] != reference_row.get('original_loss_before_source_aux',reference_row['loss']):
                raise ValueError('Initial loss differs before optimizer step')
        if active[0]:
            row['loss_support'] = {k:v for k,v in extra.items() if k!='valid'}
            # Functional autograd: does not accumulate into .grad or consume RNG.
            samples = []; seen = set()
            for n,t in named:
                group = grouping(n)
                if group not in seen: samples.append((n,t)); seen.add(group)
            components = {}
            global_grad = torch.autograd.grad(parts['global'], [t for _,t in samples], retain_graph=True, allow_unused=True)
            for key,(value,weight) in terms.items():
                gs = torch.autograd.grad(value, [t for _,t in samples], retain_graph=True, allow_unused=True)
                records = {}
                for (n,t),g,baseg in zip(samples,gs,global_grad):
                    raw = float(g.float().norm()) if g is not None else None
                    cos = None
                    if g is not None and baseg is not None:
                        den = g.float().norm()*baseg.float().norm()
                        if float(den)>0: cos=float((g.float()*baseg.float()).sum()/den)
                    records[n]={'raw_norm':raw,'weighted_norm':None if raw is None else raw*weight,'cosine_to_source_seg':cos}
                components[key]=records
            row['per_loss_representative_parameter_gradients']=components
            assert all(t.grad is None for t in params)
            if step == 1:
                # Same batch, same RNG, no hooks: prove observation did not alter loss/gradient.
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                active[0]=False; rng_set(random_state)
                with torch.cuda.amp.autocast():
                    check = legacy.loss_for_batch(model,refiner,generator,image,target,valid,lock['use_o'])[0]
                check_grads = torch.autograd.grad(check, params, allow_unused=True)
                gap = max(float((g-h).abs().max()) for g,h in zip(grads,check_grads) if g is not None and h is not None)
                row['observation_equivalence']={'loss_abs':float((loss-check).abs()), 'gradient_max_abs':gap,
                    'none_pattern_equal':all((g is None)==(h is None) for g,h in zip(grads,check_grads))}
                assert gap == 0 and float((loss-check).abs()) == 0
                del grads, check_grads, check
                rng_set(after_random)
        if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite loss')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        row['gradient_before_clip']=grad_summary(named,[t.grad for _,t in named])
        torch.nn.utils.clip_grad_norm_(refiner.parameters(),12.); torch.nn.utils.clip_grad_norm_(decoder,5.)
        row['gradient_after_clip']=grad_summary(named,[t.grad for _,t in named])
        row['gradient_optimizer_receives']=grad_summary(named,[t.grad for _,t in named])
        before={n:t.detach().clone() for n,t in named}; scale=float(scaler.get_scale())
        scaler.step(optimizer); scaler.update()
        skipped=float(scaler.get_scale())<scale; successful+=int(not skipped)
        updates={}
        for n,t in named:
            d=t.detach()-before[n]; group=grouping(n)
            u=updates.setdefault(group,{'parameter_squared':0.,'update_squared':0.,'cumulative_squared':0.,'changed':0,'elements':0,
                                        'exp_avg_squared':0.,'exp_avg_sq_sum':0.,'optimizer_steps':[]})
            u['parameter_squared']+=float(before[n].float().square().sum());u['update_squared']+=float(d.float().square().sum())
            u['cumulative_squared']+=float((t.detach().cpu()-initial[n]).float().square().sum())
            u['changed']+=int((d!=0).sum());u['elements']+=t.numel()
            state=optimizer.state[t]
            if state:
                u['optimizer_steps'].append(int(state['step']));u['exp_avg_squared']+=float(state['exp_avg'].float().square().sum())
                u['exp_avg_sq_sum']+=float(state['exp_avg_sq'].float().sum())
        for u in updates.values():
            u['parameter_norm']=u.pop('parameter_squared')**.5;u['update_norm']=u.pop('update_squared')**.5
            u['cumulative_norm']=u.pop('cumulative_squared')**.5;u['changed_fraction']=u.pop('changed')/u['elements']
            u['relative_update']=u['update_norm']/u['parameter_norm'] if u['parameter_norm']>1e-8 else None
            u['exp_avg_norm']=u.pop('exp_avg_squared')**.5
            u['optimizer_step_range']=[min(u['optimizer_steps']),max(u['optimizer_steps'])] if u['optimizer_steps'] else None
            del u['optimizer_steps']
        cases.add(batch['case'][0]);torch.cuda.synchronize();timings.append(time.monotonic()-tick)
        row.update({'actual_parameter_updates':updates,'successful_optimizer_steps':successful,'AMP_skipped':skipped,
                    'amp_scale_before':scale,'amp_scale_after':float(scaler.get_scale()),'unique_cases':len(cases),
                    'seconds':timings[-1], 'explicit_o_weight':stats(refiner.enc1.net[0].weight[:,6])})
        with (a.out/'TRAINING_TRACE.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'step':step,'loss':row['loss'],'seconds':row['seconds'],'successful':successful}),flush=True)
        del before
        if step in lock.get('intermediate_validation_steps', []):
            active[0] = False
            saved_rng = rng_get()
            saved_modes = [(module, module.training) for root in [model, refiner] for module in root.modules()]
            try:
                val = legacy.NativeWristDataset(data, 'val', False)
                validation = legacy.collect(model,refiner,generator,DataLoader(val,batch_size=1,shuffle=False,num_workers=0),device,lock['use_o'],legacy.top_training_pairs(data,15))
                write(a.out / f'VALIDATION_step{step:04d}.json', validation)
            finally:
                for module, mode in saved_modes: module.training = mode
                rng_set(saved_rng)
            print(json.dumps({'validated_step':step,'elapsed_seconds':time.monotonic()-start}),flush=True)
    for h in handles:h.remove()
    if lock['full_validation']:
        val=legacy.NativeWristDataset(data,'val',False)
        validation=legacy.collect(model,refiner,generator,DataLoader(val,batch_size=1,shuffle=False,num_workers=0),device,lock['use_o'],legacy.top_training_pairs(data,15))
        write(a.out/'FINAL_VALIDATION.json',validation)
    torch.save({'model':model.state_dict(),'refiner':refiner.state_dict(),'optimizer':optimizer.state_dict(),
                'scaler':scaler.state_dict(),'rng':rng_get(),'updates':successful},a.out/'diagnostic_final.pth')
    write(a.out/'RESULT.json',{'status':'COMPLETE','updates':successful,'attempted':len(timings),'unique_cases':len(cases),
          'seconds':time.monotonic()-start,'step_seconds':timings,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
          'test_used':False,'mode':audit['history_mode'],'lock_sha256':legacy.sha(a.lock)})


if __name__=='__main__':main()
