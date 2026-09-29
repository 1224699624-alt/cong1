"""Name-checked AdamW state port; preserves destination hyperparameters and RNG."""
from pathlib import Path
import torch


def adapt_tensor(name, source, shape):
    """Only explicitly justified R332 -> V3.1 shape changes are accepted."""
    shape = tuple(shape)
    if tuple(source.shape) == shape:
        return source.clone(), 'identity'
    if name == 'refiner.enc1.net.0.weight' and tuple(source.shape) == (12, 6, 3, 3) and shape == (12, 7, 3, 3):
        result = source.new_zeros(shape); result[:, :6] = source
        return result, 'append_zero_O_channel'
    if name.startswith('model.backbone.decoder.seg_layers.') and source.shape[0] == 15 and shape[0] == 14 and tuple(source.shape[1:]) == shape[1:]:
        return source[:14].clone(), 'keep_first14_members'
    # Frozen image-input convolution: used for verifying weights, not optimizer state.
    if name.startswith('model.backbone.encoder.stages.0.') and source.ndim == 4 and source.shape[1] == 1 and shape[1] == 2 and (source.shape[0], *source.shape[2:]) == (shape[0], *shape[2:]):
        result = source.new_zeros(shape); result[:, :1] = source
        return result, 'append_zero_seam_channel'
    raise ValueError((name, tuple(source.shape), shape))


def attach_named_state(optimizer, named_groups, source_groups, source_optimizer):
    if len(named_groups) != 2 or len(source_groups) != 2 or len(source_optimizer['param_groups']) != 2:
        raise ValueError('Expected refiner/decoder optimizer groups')
    mapped = {}; audit = []
    for group_id, (dst_named, src_named, saved_group, live_group) in enumerate(zip(named_groups, source_groups, source_optimizer['param_groups'], optimizer.param_groups)):
        if len(src_named) != len(saved_group['params']) or len(dst_named) != len(live_group['params']):
            raise ValueError('Optimizer parameter count mismatch')
        if {n for n, _ in dst_named} != {n for n, _ in src_named}:
            raise ValueError('Source/destination names differ')
        # IDs in old torch checkpoints do not contain names. Reconstruct their
        # mapping from the historical constructor + optimizer traversal, then
        # validate every state shape. Never zip old IDs directly to new params.
        by_name = {n: (param, state_id) for (n, param), state_id in zip(src_named, saved_group['params'])}
        for key in ['betas', 'eps', 'weight_decay', 'amsgrad']:
            if saved_group[key] != live_group[key]:
                raise ValueError('Unexpected hyperparameter difference: ' + key)
        for (name, param), registered in zip(dst_named, live_group['params']):
            if registered is not param: raise ValueError('Destination traversal mismatch')
            old_param, state_id = by_name[name]
            old_state = source_optimizer['state'].get(state_id)
            if not old_state or set(old_state) != {'step', 'exp_avg', 'exp_avg_sq'}:
                raise ValueError('Incomplete/unsupported Adam state for ' + name)
            state = {'step': old_state['step'].detach().cpu().clone()}
            if state['step'].numel() != 1 or float(state['step']) <= 0:
                raise ValueError('Invalid Adam step')
            change = None
            for key in ['exp_avg', 'exp_avg_sq']:
                tensor = old_state[key]
                if tensor.shape != old_param.shape: raise ValueError('Historical ID/order shape mismatch: ' + name)
                if not bool(torch.isfinite(tensor).all()): raise ValueError('Nonfinite moment: ' + name)
                if key == 'exp_avg_sq' and bool((tensor < 0).any()): raise ValueError('Negative second moment')
                new, change = adapt_tensor(name, tensor, param.shape)
                state[key] = new.to(device=param.device, dtype=param.dtype)
                if not torch.equal(new, state[key].cpu()): raise ValueError('Moment changed on transfer')
            mapped[param] = state
            audit.append({'name': name, 'group': group_id, 'source_state_id': state_id,
                          'source_shape': list(old_param.shape), 'destination_shape': list(param.shape),
                          'mapping': change, 'step': float(state['step']),
                          'exp_avg_norm': float(state['exp_avg'].norm()), 'exp_avg_sq_sum': float(state['exp_avg_sq'].sum())})
    if len(mapped) != sum(len(g['params']) for g in optimizer.param_groups):
        raise ValueError('Missing/duplicate optimizer state')
    # Commit only after all names/shapes/values are validated.
    optimizer.state.clear(); optimizer.state.update(mapped)
    return audit


def restore_r332_adam(optimizer, model, refiner, path):
    from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior
    from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner
    from train_r332_ram_joint_iterative_refinement import set_joint_trainability
    payload = torch.load(Path(path), map_location='cpu', weights_only=False)
    # Constructors consume CPU RNG, but must not alter augmentation or sampling.
    with torch.random.fork_rng(devices=[]):
        old_model = NnUNetMultiLabelPrior()
        old_refiner = InstanceCompletionRefiner()
    old_model.load_state_dict(payload['model'], strict=True)
    old_refiner.load_state_dict(payload['refiner'], strict=True)
    set_joint_trainability(old_model, True)
    old_all = {**{'model.'+n:p for n,p in old_model.named_parameters()},
               **{'refiner.'+n:p for n,p in old_refiner.named_parameters()}}
    new_all = {**{'model.'+n:p for n,p in model.named_parameters()},
               **{'refiner.'+n:p for n,p in refiner.named_parameters()}}
    for name, param in new_all.items():
        expected, _ = adapt_tensor(name, old_all[name].detach(), param.shape)
        if not torch.equal(expected, param.detach().cpu()):
            raise ValueError('Restored optimizer does not match current weights: ' + name)
    source_groups = [[('refiner.'+n,p) for n,p in old_refiner.named_parameters()],
                     [('model.'+n,p) for n,p in old_model.named_parameters() if p.requires_grad]]
    named_groups = [[('refiner.'+n,p) for n,p in refiner.named_parameters()],
                    [('model.'+n,p) for n,p in model.named_parameters() if p.requires_grad]]
    before_lrs = [g['lr'] for g in optimizer.param_groups]
    rows = attach_named_state(optimizer, named_groups, source_groups, payload['optimizer'])
    assert [g['lr'] for g in optimizer.param_groups] == before_lrs
    return {'status':'PASS', 'source':str(path), 'source_epoch':payload['epoch'], 'weights_match':True,
            'names_reconstructed_from':'historical classes and optimizer group traversal, all tensor shapes checked; checkpoint itself has no param_names',
            'restored':'AdamW exp_avg, exp_avg_sq and per-parameter step only',
            'not_restored':['historical LR=0', 'scheduler', 'GradScaler', 'RNG'],
            'destination_lrs_unchanged':before_lrs, 'parameters':rows}
