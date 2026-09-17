"""R336-TSRS: state-masked seam loss with logit PCGrad and selective distillation."""
from __future__ import annotations

import copy, json, os, sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

# The runner exports the repository scripts directory.
from r336_gradient_surgery import gradient_surrogate, selective_softmax_distillation


class nnUNetTrainerR336StatePCGradDistill(nnUNetTrainer):
    def __init__(self, plans:dict, configuration:str, fold:int, dataset_json:dict,
                 device:torch.device=torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        if self.plans_manager.dataset_name!='Dataset204_TSRS_RSNAEpiphysisMaturePrior2D':raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr=3e-5;self.num_epochs=int(os.environ.get('R336_TSRS_EPOCHS','18'))
        self.seam_weight=float(os.environ.get('R336_TSRS_SEAM_WEIGHT','.020'));self.distill_weight=float(os.environ.get('R336_TSRS_DISTILL_WEIGHT','.20'));self.ceiling=.10
        self.early_min_epochs=12;self.early_patience=6;self.early_min_delta=1e-4;self._early_best=None;self._early_bad=0;self._early_stop=False;self.teacher=None
        np.random.seed(3362);torch.manual_seed(3362);torch.cuda.manual_seed_all(3362)

    def initialize(self):
        if self.was_initialized:return
        super().initialize();path=Path(os.environ['R336_TSRS_INIT_CHECKPOINT']);payload=torch.load(path,map_location=self.device,weights_only=False)
        self.network.load_state_dict(payload['network_weights'],strict=True);self.teacher=copy.deepcopy(self.network).eval().requires_grad_(False)
        (Path(self.output_folder)/'r336_initialization.json').write_text(json.dumps({'checkpoint':str(path),'strict':True,'teacher_frozen':True},indent=2))

    def configure_optimizers(self):
        opt=torch.optim.SGD(self.network.parameters(),self.initial_lr,weight_decay=self.weight_decay,momentum=.99,nesterov=True)
        return opt,PolyLRScheduler(opt,self.initial_lr,self.num_epochs)

    def on_train_epoch_start(self):
        super().on_train_epoch_start();self.teacher.eval()

    def train_step(self,batch):
        data=batch['data'].to(self.device,non_blocking=True);target=batch['target'];target=[x.to(self.device,non_blocking=True) for x in target] if isinstance(target,list) else target.to(self.device,non_blocking=True)
        high=target[0] if isinstance(target,list) else target;self.optimizer.zero_grad(set_to_none=True);ctx=autocast(self.device.type,enabled=True) if self.device.type=='cuda' else nullcontext()
        with ctx:
            output=self.network(data);base=self.loss(output,target);logits=output[0] if isinstance(output,(list,tuple)) else output
            with torch.no_grad():tout=self.teacher(data);teacher_logits=tout[0] if isinstance(tout,(list,tuple)) else tout
            prior=data[:,1].float();lo=prior.amin((1,2),keepdim=True);hi=prior.amax((1,2),keepdim=True);prior=(prior-lo)/(hi-lo+1e-6)
            foreground=(high[:,0]>0).float();seam=prior.square()*(1-foreground);prob=torch.softmax(logits.float(),dim=1)[:,1]
            seam_loss=((seam*F.relu(prob-self.ceiling).square()).flatten(1).sum(1)/seam.flatten(1).sum(1).clamp_min(1)).mean()
            seam_surrogate,stats=gradient_surrogate(logits,seam_loss,base,self.seam_weight)
            uncertain=(seam<=0).float();distill,distill_mass=selective_softmax_distillation(logits,teacher_logits,high[:,0],uncertain,.80)
            loss=base+seam_surrogate+self.distill_weight*distill
        if self.grad_scaler is not None:self.grad_scaler.scale(loss).backward();self.grad_scaler.unscale_(self.optimizer)
        else:loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),12)
        if self.grad_scaler is not None:self.grad_scaler.step(self.optimizer);self.grad_scaler.update()
        else:self.optimizer.step()
        return {'loss':loss.detach().cpu().numpy(),'base_loss':base.detach().cpu().numpy(),'seam_loss':seam_loss.detach().cpu().numpy(),'distill_loss':distill.detach().cpu().numpy(),'conflict_fraction':np.asarray(stats['conflict_fraction'],dtype=np.float32),'distill_mask_mass':np.asarray(distill_mass,dtype=np.float32)}

    def on_train_epoch_end(self,outputs):
        super().on_train_epoch_end(outputs);row={k:float(np.mean([float(np.asarray(x[k])) for x in outputs])) for k in outputs[0] if k!='loss'};row['epoch']=int(self.current_epoch)
        (Path(self.output_folder)/'r336_dynamics.jsonl').open('a').write(json.dumps(row)+'\n')

    def on_epoch_end(self):
        super().on_epoch_end();score=float(self.logger.get_value('ema_fg_dice',step=-1));improved=self._early_best is None or score>self._early_best+self.early_min_delta
        if improved:self._early_best=score;self._early_bad=0
        else:self._early_bad+=1
        if self.current_epoch>=self.early_min_epochs and self._early_bad>=self.early_patience:self._early_stop=True
        (Path(self.output_folder)/'r336_early_stop.json').write_text(json.dumps({'completed_epochs':int(self.current_epoch),'ema_dice':score,'best':self._early_best,'bad_epochs':self._early_bad,'stop':self._early_stop},indent=2))

    def run_training(self):
        self.on_train_start()
        while self.current_epoch<self.num_epochs and not self._early_stop:
            self.on_epoch_start();self.on_train_epoch_start();train=[self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)];self.on_train_epoch_end(train)
            with torch.no_grad():self.on_validation_epoch_start();val=[self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)];self.on_validation_epoch_end(val)
            self.on_epoch_end()
        self.on_train_end()
