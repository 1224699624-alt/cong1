"""R335-TSRS loss-only conditional seam-state trainer.

The R317 architecture and two-channel input remain unchanged. TSRS supplies a
reliable binary background seam state but no trustworthy projection-overlap
state; uncertain pixels receive only the ordinary nnU-Net segmentation loss.
"""
from __future__ import annotations

import json, os
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


class nnUNetTrainerR335ConditionalStateLoss(nnUNetTrainer):
    def __init__(self, plans:dict, configuration:str, fold:int, dataset_json:dict,
                 device:torch.device=torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        expected='Dataset204_TSRS_RSNAEpiphysisMaturePrior2D'
        if self.plans_manager.dataset_name != expected: raise RuntimeError(f'{self.plans_manager.dataset_name} != {expected}')
        self.initial_lr=5e-5; self.num_epochs=int(os.environ.get('R335_TSRS_EPOCHS','18'))
        self.seam_weight=float(os.environ.get('R335_TSRS_SEAM_WEIGHT','.020'))
        self.support_weight=float(os.environ.get('R335_TSRS_SUPPORT_WEIGHT','.010'))
        self.ceiling=float(os.environ.get('R335_TSRS_SEAM_CEILING','.10'))
        self.early_min_epochs=9;self.early_patience=6;self.early_min_delta=1e-4
        self._early_best=None;self._early_bad=0;self._early_stop=False
        np.random.seed(3352);torch.manual_seed(3352);torch.cuda.manual_seed_all(3352)

    def initialize(self):
        if self.was_initialized:return
        nnUNetTrainer.initialize(self)
        path=Path(os.environ['R335_TSRS_INIT_CHECKPOINT']);payload=torch.load(path,map_location=self.device,weights_only=False)
        state=payload['network_weights'];self.network.load_state_dict(state,strict=True)
        first=state['encoder.stages.0.0.convs.0.conv.weight']
        if tuple(first.shape)!=(32,2,3,3):raise RuntimeError(first.shape)
        (Path(self.output_folder)/'r335_initialization.json').write_text(json.dumps({'checkpoint':str(path),'strict':True,'prior_channel_nonzero':int(torch.count_nonzero(first[:,1]))},indent=2))

    def configure_optimizers(self):
        opt=torch.optim.SGD(self.network.parameters(),self.initial_lr,weight_decay=self.weight_decay,momentum=.99,nesterov=True)
        return opt,PolyLRScheduler(opt,self.initial_lr,self.num_epochs)

    def train_step(self,batch):
        data=batch['data'].to(self.device,non_blocking=True);target=batch['target'];target=[x.to(self.device,non_blocking=True) for x in target] if isinstance(target,list) else target.to(self.device,non_blocking=True)
        high=target[0] if isinstance(target,list) else target;self.optimizer.zero_grad(set_to_none=True);ctx=autocast(self.device.type,enabled=True) if self.device.type=='cuda' else nullcontext()
        with ctx:
            output=self.network(data);base=self.loss(output,target);logits=output[0] if isinstance(output,(list,tuple)) else output
            prior=data[:,1].float();lo=prior.amin((1,2),keepdim=True);hi=prior.amax((1,2),keepdim=True);prior=(prior-lo)/(hi-lo+1e-6)
            foreground=(high[:,0]>0).float();seam=prior.square()*(1-foreground)
            probability=torch.softmax(logits.float(),dim=1)[:,1]
            seam_loss=(seam*F.relu(probability-self.ceiling).square()).flatten(1).sum(1)/seam.flatten(1).sum(1).clamp_min(1)
            seam_loss=seam_loss.mean()
            # A high-confidence eroded foreground state protects bone interiors from contraction.
            support=1-F.max_pool2d((1-foreground)[:,None],5,1,2)[:,0]
            support_loss=(support*F.relu(.80-probability).square()).flatten(1).sum(1)/support.flatten(1).sum(1).clamp_min(1)
            support_loss=support_loss.mean();loss=base+self.seam_weight*seam_loss+self.support_weight*support_loss
        if self.grad_scaler is not None:self.grad_scaler.scale(loss).backward();self.grad_scaler.unscale_(self.optimizer)
        else:loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),12)
        if self.grad_scaler is not None:self.grad_scaler.step(self.optimizer);self.grad_scaler.update()
        else:self.optimizer.step()
        return {'loss':loss.detach().cpu().numpy(),'base_loss':base.detach().cpu().numpy(),'seam_loss':seam_loss.detach().cpu().numpy(),'support_loss':support_loss.detach().cpu().numpy(),'seam_mass':seam.mean().detach().cpu().numpy(),'support_mass':support.mean().detach().cpu().numpy(),'overlap_state_mass':np.asarray(0.,dtype=np.float32)}

    def on_train_epoch_end(self,outputs):
        super().on_train_epoch_end(outputs);row={k:float(np.mean([float(np.asarray(x[k])) for x in outputs])) for k in outputs[0] if k!='loss'};row['epoch']=int(self.current_epoch)
        (Path(self.output_folder)/'r335_conditional_dynamics.jsonl').open('a').write(json.dumps(row)+'\n')

    def on_epoch_end(self):
        super().on_epoch_end();score=float(self.logger.get_value('ema_fg_dice',step=-1));improved=self._early_best is None or score>self._early_best+self.early_min_delta
        if improved:self._early_best=score;self._early_bad=0
        else:self._early_bad+=1
        if self.current_epoch>=self.early_min_epochs and self._early_bad>=self.early_patience:self._early_stop=True
        (Path(self.output_folder)/'r335_early_stop.json').write_text(json.dumps({'completed_epochs':int(self.current_epoch),'ema_dice':score,'best':self._early_best,'bad_epochs':self._early_bad,'stop':self._early_stop},indent=2))

    def run_training(self):
        self.on_train_start()
        while self.current_epoch<self.num_epochs and not self._early_stop:
            self.on_epoch_start();self.on_train_epoch_start();train=[self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)];self.on_train_epoch_end(train)
            with torch.no_grad():self.on_validation_epoch_start();val=[self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)];self.on_validation_epoch_end(val)
            self.on_epoch_end()
        self.on_train_end()
