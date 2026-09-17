"""Matched nnU-Net fork for the fixed R258B prior visual probe."""
from __future__ import annotations
import json
from contextlib import nullcontext
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

class _Base(nnUNetTrainer):
 def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
  super().__init__(plans,configuration,fold,dataset_json,device)
  if self.plans_manager.dataset_name!='Dataset203_TSRS_RSNAEpiphysisPrior2D': raise RuntimeError(self.plans_manager.dataset_name)
  self.initial_lr=1e-4; self.num_iterations_per_epoch=25; self.num_val_iterations_per_epoch=10; np.random.seed(259); torch.manual_seed(259); torch.cuda.manual_seed_all(259)
 def configure_optimizers(self):
  opt=torch.optim.SGD(self.network.parameters(),self.initial_lr,weight_decay=self.weight_decay,momentum=.99,nesterov=True); return opt,PolyLRScheduler(opt,self.initial_lr,4)

class _ZeroPrior(_Base):
 def train_step(self,batch):
  batch=dict(batch); batch['data']=batch['data'].clone(); batch['data'][:,1]=0; return super().train_step(batch)

class nnUNetTrainerR259Warmup(_ZeroPrior):
 def __init__(self,plans:dict,configuration:str,fold:int,dataset_json:dict,device:torch.device=torch.device('cuda')): super().__init__(plans,configuration,fold,dataset_json,device); self.num_epochs=2
class nnUNetTrainerR259Native(_ZeroPrior):
 def __init__(self,plans:dict,configuration:str,fold:int,dataset_json:dict,device:torch.device=torch.device('cuda')): super().__init__(plans,configuration,fold,dataset_json,device); self.num_epochs=4

class nnUNetTrainerR259FrozenPrior(_Base):
 def __init__(self,plans:dict,configuration:str,fold:int,dataset_json:dict,device:torch.device=torch.device('cuda')): super().__init__(plans,configuration,fold,dataset_json,device); self.num_epochs=4; self.prior_alpha=.05
 def train_step(self,batch):
  data=batch['data'].to(self.device,non_blocking=True); target=batch['target']; target=[x.to(self.device,non_blocking=True) for x in target] if isinstance(target,list) else target.to(self.device,non_blocking=True); high=target[0] if isinstance(target,list) else target; self.optimizer.zero_grad(set_to_none=True)
  context=autocast(self.device.type,enabled=True) if self.device.type=='cuda' else nullcontext()
  with context:
   output=self.network(data); base=self.loss(output,target); logits=output[0] if isinstance(output,(list,tuple)) else output; prior=data[:,1].float(); lo=prior.amin((1,2),keepdim=True); hi=prior.amax((1,2),keepdim=True); prior=(prior-lo)/(hi-lo+1e-6); background=(high[:,0]==0).float(); weight=prior.square()*background; log_odds=logits[:,1].float()-logits[:,0].float(); numer=(weight*F.softplus(log_odds)).flatten(1).sum(1); denom=weight.flatten(1).sum(1); valid=denom>1e-6; pair=(numer[valid]/(denom[valid]+1e-6)).mean() if valid.any() else logits.sum()*0.; loss=base+self.prior_alpha*pair
  if self.grad_scaler is not None: self.grad_scaler.scale(loss).backward(); self.grad_scaler.unscale_(self.optimizer)
  else: loss.backward()
  torch.nn.utils.clip_grad_norm_(self.network.parameters(),12)
  if self.grad_scaler is not None: self.grad_scaler.step(self.optimizer); self.grad_scaler.update()
  else:self.optimizer.step()
  return {'loss':loss.detach().cpu().numpy(),'base_loss':base.detach().cpu().numpy(),'prior_loss':pair.detach().cpu().numpy(),'prior_mass':weight.mean().detach().cpu().numpy()}
 def on_train_epoch_end(self,outputs):
  super().on_train_epoch_end(outputs); row={k:float(np.mean([float(np.asarray(x[k])) for x in outputs])) for k in outputs[0] if k!='loss'}; row['epoch']=int(self.current_epoch); p=Path(self.output_folder)/'r259_prior_dynamics.jsonl'; p.open('a').write(json.dumps(row)+'\n'); self.print_to_log_file('R259 dynamics',json.dumps(row))
