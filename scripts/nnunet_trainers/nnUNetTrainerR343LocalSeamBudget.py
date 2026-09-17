"""R343: TSRS local-interface adaptive seam loss.

R342 used one scalar weight per image. R343 computes a separate bounded weight
for every pixel in each predicted seam interface, so distinct gaps in one hand
can be corrected or ignored independently. The GT-derived benefit is training
only; inference uses only the frozen R317 prior channel and current logits.
"""
from __future__ import annotations
import copy,json,os
from contextlib import nullcontext
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

class nnUNetTrainerR343LocalSeamBudget(nnUNetTrainer):
    def __init__(self,plans:dict,configuration:str,fold:int,dataset_json:dict,device:torch.device=torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        if self.plans_manager.dataset_name!='Dataset204_TSRS_RSNAEpiphysisMaturePrior2D':raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr=float(os.environ.get('R343_LR','2e-5'));self.num_epochs=int(os.environ.get('R343_EPOCHS','18'))
        self.budget=float(os.environ.get('R343_BUDGET','.10'));self.preserve_weight=float(os.environ.get('R343_PRESERVE','.10'))
        self.ceiling=float(os.environ.get('R343_CEILING','.10'));self.edit_strength=float(os.environ.get('R343_EDIT','.25'))
        self.teacher=None;np.random.seed(3431);torch.manual_seed(3431);torch.cuda.manual_seed_all(3431)

    def initialize(self):
        if self.was_initialized:return
        super().initialize();path=Path(os.environ['R343_INIT_CHECKPOINT']);payload=torch.load(path,map_location=self.device,weights_only=False)
        self.network.load_state_dict(payload['network_weights'],strict=True);self.teacher=copy.deepcopy(self.network).eval().requires_grad_(False)
        (Path(self.output_folder)/'r343_initialization.json').write_text(json.dumps({'checkpoint':str(path),'strict':True,'teacher_frozen':True,'local_weight_map':True},indent=2))

    def configure_optimizers(self):
        opt=torch.optim.SGD(self.network.parameters(),self.initial_lr,weight_decay=self.weight_decay,momentum=.99,nesterov=True)
        return opt,PolyLRScheduler(opt,self.initial_lr,self.num_epochs)

    def on_train_epoch_start(self):super().on_train_epoch_start();self.teacher.eval()

    def train_step(self,batch):
        data=batch['data'].to(self.device,non_blocking=True);target=batch['target'];target=[x.to(self.device,non_blocking=True) for x in target] if isinstance(target,list) else target.to(self.device,non_blocking=True)
        high=target[0] if isinstance(target,list) else target;fg=(high[:,0]>0).float();self.optimizer.zero_grad(set_to_none=True)
        ctx=autocast(self.device.type,enabled=True) if self.device.type=='cuda' else nullcontext()
        with ctx:
            output=self.network(data);base=self.loss(output,target);logits=output[0] if isinstance(output,(list,tuple)) else output
            with torch.no_grad():tout=self.teacher(data);teacher_logits=tout[0] if isinstance(tout,(list,tuple)) else tout
            prior=data[:,1].float();lo=prior.amin((1,2),keepdim=True);hi=prior.amax((1,2),keepdim=True);prior=((prior-lo)/(hi-lo+1e-6)).clamp(0,1)
            p=torch.softmax(logits.float(),dim=1)[:,1];seam=prior.square()*(1-fg)
            # Local utility of a bounded foreground reduction at each seam pixel.
            edited=(p-self.edit_strength*seam*p).clamp(1e-5,1-1e-5);ps=p.clamp(1e-5,1-1e-5)
            e0=-(fg*torch.log(ps)+(1-fg)*torch.log1p(-ps));e1=-(fg*torch.log(edited)+(1-fg)*torch.log1p(-edited))
            benefit=torch.sigmoid((e0-e1)/.002)
            uncertainty=4*p*(1-p);reliability=prior*(.25+.75*uncertainty)
            local_gate=(benefit*reliability).clamp(0,1)
            # Per-pixel budget: no seam pixel can contribute more than 10% of
            # the corresponding primary pixel loss scale.
            interaction=(seam*F.relu(p-self.ceiling).square())
            # Explicit probability-form CE is AMP-safe here (unlike
            # torch.nn.functional.binary_cross_entropy on CUDA autocast).
            primary=(-(fg*torch.log(p)+(1-fg)*torch.log1p(-p))).clamp_min(1e-5)
            primary=torch.nan_to_num(primary,nan=1e-5,posinf=1e5,neginf=1e-5)
            local_gate=torch.nan_to_num(local_gate,nan=0.0,posinf=1.0,neginf=0.0)
            interaction=torch.nan_to_num(interaction,nan=0.0,posinf=1e3,neginf=0.0)
            safe=torch.minimum(local_gate,self.budget*primary.detach()/(interaction.detach()+1e-5))
            safe=torch.nan_to_num(safe,nan=0.0,posinf=1.0,neginf=0.0).clamp(0,1)
            seam_loss=(safe*interaction).sum((1,2))/(safe*seam).sum((1,2)).clamp_min(1e-5)
            seam_loss=torch.nan_to_num(seam_loss,nan=0.0,posinf=1e3,neginf=0.0)
            support=1-F.max_pool2d((1-fg)[:,None],5,1,2)[:,0];support_err=F.relu(.80-p).square()
            support_loss=(support*support_err).flatten(1).sum(1)/support.flatten(1).sum(1).clamp_min(1)
            teacher_prob=torch.softmax(teacher_logits.float(),dim=1);conf=teacher_prob.max(1).values
            preserve_mask=(seam<.05).float()*(conf>.90).float();kl=F.kl_div(torch.log_softmax(logits.float(),dim=1),teacher_prob,reduction='none').sum(1)
            preserve=(kl*preserve_mask).flatten(1).sum(1)/preserve_mask.flatten(1).sum(1).clamp_min(1)
            loss=base+seam_loss.mean()+.01*support_loss.mean()+self.preserve_weight*preserve.mean()
        if self.grad_scaler is not None:self.grad_scaler.scale(loss).backward();self.grad_scaler.unscale_(self.optimizer)
        else:loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),12)
        if self.grad_scaler is not None:self.grad_scaler.step(self.optimizer);self.grad_scaler.update()
        else:self.optimizer.step()
        active=(safe*seam>.01).float();return {'loss':loss.detach().cpu().numpy(),'base_loss':base.detach().cpu().numpy(),'seam_loss':seam_loss.mean().detach().cpu().numpy(),'support_loss':support_loss.mean().detach().cpu().numpy(),'preserve_loss':preserve.mean().detach().cpu().numpy(),'local_gate_mean':local_gate.mean().detach().cpu().numpy(),'local_gate_std':local_gate.std(unbiased=False).detach().cpu().numpy(),'safe_weight_mean':safe.mean().detach().cpu().numpy(),'safe_weight_std':safe.std(unbiased=False).detach().cpu().numpy(),'active_gap_fraction':active.mean().detach().cpu().numpy(),'benefit_mean':benefit.mean().detach().cpu().numpy(),'seam_mass':seam.mean().detach().cpu().numpy()}

    def on_train_epoch_end(self,outputs):
        super().on_train_epoch_end(outputs);row={k:float(np.mean([float(np.asarray(x[k])) for x in outputs])) for k in outputs[0] if k!='loss'};row['epoch']=int(self.current_epoch);(Path(self.output_folder)/'r343_local_dynamics.jsonl').open('a').write(json.dumps(row)+'\n')

    def run_training(self):
        self.on_train_start()
        while self.current_epoch<self.num_epochs:
            self.on_epoch_start();self.on_train_epoch_start();train=[self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)];self.on_train_epoch_end(train)
            with torch.no_grad():self.on_validation_epoch_start();val=[self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)];self.on_validation_epoch_end(val)
            self.on_epoch_end()
        self.on_train_end()
