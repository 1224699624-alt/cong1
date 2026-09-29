"""Training-only direct source supervision for unselected hard overlap FN."""
import torch
import torch.nn.functional as F


def local_negative_guard_loss(source,target,valid,teacher_final,denominator,band=.25):
    """Training GT-negative near-decision region from a frozen common-start teacher.

    The teacher is a region selector only. Neither validation errors nor student
    predictions select this mask; the original positive auxiliary is unchanged.
    """
    if source.shape != target.shape or target.shape[1] != 14 or teacher_final.shape != source.shape:
        raise ValueError('complete RAM14 contract required')
    if denominator <= 0 or band <= 0:
        raise ValueError('positive denominator and band required')
    support=(target<.5)&valid.bool()&(teacher_final.detach().float().abs()<band)
    coefficient=2.2/denominator
    loss=(F.softplus(source.float())*support).sum()*coefficient
    return loss,{'negative_members':int(support.sum()),'denominator':float(denominator),
                 'per_pixel_coefficient':float(coefficient),'weighted_loss':float(loss.detach()),
                 'teacher_band':band,'source_only':True,'teacher_frozen':True,
                 'GT_used_only_in_training_loss':True}


def hard_source_loss(source,target,valid,indices):
    if target.shape[1]!=14 or source.shape!=target.shape:raise ValueError('complete RAM14 contract required')
    with torch.no_grad():
        probability=source.sigmoid()
        selected_target=target[0,indices].unsqueeze(1)
        initial_error=(probability[0,indices].unsqueeze(1)-selected_target).abs()
        local=F.max_pool2d((initial_error>.15).float(),11,1,5)
        denominator=(1+3*local).sum()
        selected=torch.zeros_like(target,dtype=torch.bool);selected[:,indices]=True
        true_overlap=target.sum(1,keepdim=True)>=2
        hard=(target>.5)&true_overlap&valid.bool()&(source.detach().float() < -5.00001)
        support=hard&~selected
    # Matches the direct per-pixel selected completion BCE coefficient at local=1:
    # sum of step weights=1; BCE weight=.55; local pixel weight=4.
    coefficient=2.2/denominator
    loss=(F.softplus(-source.float())*support).sum()*coefficient
    return loss,{'hard_members':int(hard.sum()),'unselected_hard_members':int(support.sum()),
                 'selected_hard_members':int((hard&selected).sum()),'denominator':float(denominator),
                 'per_pixel_coefficient':float(coefficient),'weighted_loss':float(loss.detach()),
                 'source_only':True,'GT_used_only_in_training_loss':True}
