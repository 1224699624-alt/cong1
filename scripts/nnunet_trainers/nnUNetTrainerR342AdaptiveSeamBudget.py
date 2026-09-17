"""R342: TSRS sample-adaptive seam loss with safe budget and R317 preservation."""
from __future__ import annotations
import copy,json,os
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


class nnUNetTrainerR342AdaptiveSeamBudget(nnUNetTrainer):
    def __init__(self,plans:dict,configuration:str,fold:int,dataset_json:dict,device:torch.device=torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        if self.plans_manager.dataset_name!='Dataset204_TSRS_RSNAEpiphysisMaturePrior2D':raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr=float(os.environ.get('R342_LR','2e-5'));self.num_epochs=int(os.environ.get('R342_EPOCHS','18'))
        self.interaction_budget=float(os.environ.get('R342_BUDGET','.10'));self.preserve_weight=float(os.environ.get('R342_PRESERVE','.10'))
        self.seam_ceiling=float(os.environ.get('R342_CEILING','.10'));self.edit_strength=float(os.environ.get('R342_EDIT','.25'))
        self.teacher=None;np.random.seed(3421);torch.manual_seed(3421);torch.cuda.manual_seed_all(3421)

    def initialize(self):
        if self.was_initialized:return
        super().initialize();path=Path(os.environ['R342_INIT_CHECKPOINT']);payload=torch.load(path,map_location=self.device,weights_only=False)
        self.network.load_state_dict(payload['network_weights'],strict=True);self.teacher=copy.deepcopy(self.network).eval().requires_grad_(False)
        (Path(self.output_folder)/'r342_initialization.json').write_text(json.dumps({'checkpoint':str(path),'strict':True,'teacher_frozen':True,
          'fixed_epochs_no_patience_stop':self.num_epochs,'interaction_budget':self.interaction_budget},indent=2))

    def configure_optimizers(self):
        opt=torch.optim.SGD(self.network.parameters(),self.initial_lr,weight_decay=self.weight_decay,momentum=.99,nesterov=True)
        return opt,PolyLRScheduler(opt,self.initial_lr,self.num_epochs)

    def on_train_epoch_start(self):super().on_train_epoch_start();self.teacher.eval()

    def train_step(self,batch):
        data=batch['data'].to(self.device,non_blocking=True);target=batch['target'];target=[x.to(self.device,non_blocking=True) for x in target] if isinstance(target,list) else target.to(self.device,non_blocking=True)
        high=target[0] if isinstance(target,list) else target;foreground=(high[:,0]>0).float();self.optimizer.zero_grad(set_to_none=True)
        ctx=autocast(self.device.type,enabled=True) if self.device.type=='cuda' else nullcontext()
        with ctx:
            output=self.network(data);base=self.loss(output,target);logits=output[0] if isinstance(output,(list,tuple)) else output
            with torch.no_grad():tout=self.teacher(data);teacher_logits=tout[0] if isinstance(tout,(list,tuple)) else tout
            prior=data[:,1].float();lo=prior.amin((1,2),keepdim=True);hi=prior.amax((1,2),keepdim=True);prior=(prior-lo)/(hi-lo+1e-6)
            seam=prior.square()*(1-foreground);p=torch.softmax(logits.float(),dim=1)[:,1]
            seam_per=(seam*F.relu(p-self.seam_ceiling).square()).flatten(1).sum(1)/seam.flatten(1).sum(1).clamp_min(1)
            support=1-F.max_pool2d((1-foreground)[:,None],5,1,2)[:,0]
            support_per=(support*F.relu(.80-p).square()).flatten(1).sum(1)/support.flatten(1).sum(1).clamp_min(1)
            interaction=seam_per+.5*support_per
            # Training-only utility of a bounded seam correction. GT never enters inference.
            with torch.no_grad():
                edited=(p-self.edit_strength*seam*p).clamp(1e-5,1-1e-5)
                p_safe=p.clamp(1e-5,1-1e-5);edited_safe=edited.clamp(1e-5,1-1e-5)
                e0=(-(foreground*torch.log(p_safe)+(1-foreground)*torch.log1p(-p_safe))).flatten(1).mean(1)
                e1=(-(foreground*torch.log(edited_safe)+(1-foreground)*torch.log1p(-edited_safe))).flatten(1).mean(1)
                benefit=torch.sigmoid((e0-e1)/.002)
                uncertainty=((4*p*(1-p)*seam).flatten(1).sum(1)/seam.flatten(1).sum(1).clamp_min(1))
                reliability=(seam.flatten(1).sum(1)/(prior.square().flatten(1).sum(1)+1)).clamp(0,1)
                sample_gate=benefit*reliability*(.25+.75*uncertainty)
                budget=(self.interaction_budget*e0/(interaction.detach()+1e-6)).clamp(max=self.interaction_budget)
                safe_weight=torch.minimum(sample_gate,budget)
            teacher_prob=torch.softmax(teacher_logits.float(),dim=1);teacher_conf=teacher_prob.max(1).values
            noninteraction=(seam<.05).float();preserve_mask=noninteraction*(teacher_conf>.90).float()
            kl=F.kl_div(torch.log_softmax(logits.float(),dim=1),teacher_prob,reduction='none').sum(1)
            preserve_per=(kl*preserve_mask).flatten(1).sum(1)/preserve_mask.flatten(1).sum(1).clamp_min(1)
            adaptive=(safe_weight*interaction).mean();preserve=preserve_per.mean();loss=base+adaptive+self.preserve_weight*preserve
        if self.grad_scaler is not None:self.grad_scaler.scale(loss).backward();self.grad_scaler.unscale_(self.optimizer)
        else:loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),12)
        if self.grad_scaler is not None:self.grad_scaler.step(self.optimizer);self.grad_scaler.update()
        else:self.optimizer.step()
        return {'loss':loss.detach().cpu().numpy(),'base_loss':base.detach().cpu().numpy(),'adaptive_loss':adaptive.detach().cpu().numpy(),
          'seam_loss':seam_per.mean().detach().cpu().numpy(),'support_loss':support_per.mean().detach().cpu().numpy(),'preserve_loss':preserve.detach().cpu().numpy(),
          'sample_gate_mean':sample_gate.mean().detach().cpu().numpy(),'sample_gate_std':sample_gate.std(unbiased=False).detach().cpu().numpy(),
          'safe_weight_mean':safe_weight.mean().detach().cpu().numpy(),'safe_weight_std':safe_weight.std(unbiased=False).detach().cpu().numpy(),
          'benefit_mean':benefit.mean().detach().cpu().numpy(),'reliability_mean':reliability.mean().detach().cpu().numpy(),'seam_mass':seam.mean().detach().cpu().numpy()}

    def on_train_epoch_end(self,outputs):
        super().on_train_epoch_end(outputs);row={k:float(np.mean([float(np.asarray(x[k])) for x in outputs])) for k in outputs[0] if k!='loss'};row['epoch']=int(self.current_epoch)
        (Path(self.output_folder)/'r342_adaptive_dynamics.jsonl').open('a').write(json.dumps(row)+'\n')

    def run_training(self):
        self.on_train_start()
        while self.current_epoch<self.num_epochs:
            self.on_epoch_start();self.on_train_epoch_start();train=[self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)];self.on_train_epoch_end(train)
            with torch.no_grad():self.on_validation_epoch_start();val=[self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)];self.on_validation_epoch_end(val)
            self.on_epoch_end()
        self.on_train_end()
