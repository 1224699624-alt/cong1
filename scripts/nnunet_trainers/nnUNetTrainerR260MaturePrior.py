"""Mature R202-initialized two-channel prior trainer with validation early stopping."""
from __future__ import annotations
import json,os,hashlib
from contextlib import nullcontext
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
class nnUNetTrainerR260MaturePrior(nnUNetTrainer):
 def __init__(self,plans:dict,configuration:str,fold:int,dataset_json:dict,device:torch.device=torch.device('cuda')):
  super().__init__(plans,configuration,fold,dataset_json,device)
  expected=os.environ.get('R315_EXPECTED_DATASET','Dataset204_TSRS_RSNAEpiphysisMaturePrior2D')
  if self.plans_manager.dataset_name!=expected: raise RuntimeError(f'{self.plans_manager.dataset_name} != {expected}')
  self.initial_lr=1e-4; self.num_epochs=60; self.prior_alpha=.05; self.early_min_epochs=15; self.early_patience=10; self.early_min_delta=1e-4; self._early_best=None; self._early_bad=0; self._early_stop=False; np.random.seed(260); torch.manual_seed(260); torch.cuda.manual_seed_all(260)
 def initialize(self):
  if self.was_initialized:return
  if os.environ.get('nnUNet_compile','').lower()!='false': raise RuntimeError('R260 requires nnUNet_compile=false for strict R202 loading')
  super().initialize(); path=Path(os.environ['R260_ADAPTED_CHECKPOINT']); payload=torch.load(path,map_location=self.device,weights_only=False); state=payload['network_weights']; self.network.load_state_dict(state,strict=True)
  first=state['encoder.stages.0.0.convs.0.conv.weight']; assert tuple(first.shape)==(32,2,3,3) and int(torch.count_nonzero(first[:,1]))==0
  heads=[k for k in state if 'seg_layers' in k]
  if not heads: raise RuntimeError('No inherited segmentation heads found')
  manifest={'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'strict_all_network_keys_loaded':True,'prior_channel_nonzero':0,'segmentation_head_tensors_loaded':len(heads)}
  (Path(self.output_folder)/'r260_mature_initialization.json').write_text(json.dumps(manifest,indent=2)); self.print_to_log_file('R260 mature initialization',json.dumps(manifest))
 def configure_optimizers(self):
  opt=torch.optim.SGD(self.network.parameters(),self.initial_lr,weight_decay=self.weight_decay,momentum=.99,nesterov=True); return opt,PolyLRScheduler(opt,self.initial_lr,60)
 def train_step(self,batch):
  data=batch['data'].to(self.device,non_blocking=True); target=batch['target']; target=[x.to(self.device,non_blocking=True) for x in target] if isinstance(target,list) else target.to(self.device,non_blocking=True); high=target[0] if isinstance(target,list) else target; self.optimizer.zero_grad(set_to_none=True); context=autocast(self.device.type,enabled=True) if self.device.type=='cuda' else nullcontext()
  with context:
   output=self.network(data); base=self.loss(output,target); logits=output[0] if isinstance(output,(list,tuple)) else output; prior=data[:,1].float(); lo=prior.amin((1,2),keepdim=True); hi=prior.amax((1,2),keepdim=True); prior=(prior-lo)/(hi-lo+1e-6); weight=prior.square()*(high[:,0]==0).float(); odds=logits[:,1].float()-logits[:,0].float(); num=(weight*F.softplus(odds)).flatten(1).sum(1); den=weight.flatten(1).sum(1); valid=den>1e-6; pair=(num[valid]/(den[valid]+1e-6)).mean() if valid.any() else logits.sum()*0; loss=base+self.prior_alpha*pair
  if self.grad_scaler is not None:self.grad_scaler.scale(loss).backward(); self.grad_scaler.unscale_(self.optimizer)
  else:loss.backward()
  torch.nn.utils.clip_grad_norm_(self.network.parameters(),12)
  if self.grad_scaler is not None:self.grad_scaler.step(self.optimizer); self.grad_scaler.update()
  else:self.optimizer.step()
  return {'loss':loss.detach().cpu().numpy(),'base_loss':base.detach().cpu().numpy(),'prior_loss':pair.detach().cpu().numpy(),'prior_mass':weight.mean().detach().cpu().numpy()}
 def on_train_epoch_end(self,outputs):
  super().on_train_epoch_end(outputs); row={k:float(np.mean([float(np.asarray(x[k])) for x in outputs])) for k in outputs[0] if k!='loss'}; row['epoch']=int(self.current_epoch); (Path(self.output_folder)/'r260_prior_dynamics.jsonl').open('a').write(json.dumps(row)+'\n')
 def on_epoch_end(self):
  super().on_epoch_end(); score=float(self.logger.get_value('ema_fg_dice',step=-1)); improved=self._early_best is None or score>self._early_best+self.early_min_delta
  if improved:self._early_best=score; self._early_bad=0
  else:self._early_bad+=1
  state={'completed_epochs':int(self.current_epoch),'ema_dice':score,'early_best':self._early_best,'bad_epochs':self._early_bad,'stop':False}
  if self.current_epoch>=self.early_min_epochs and self._early_bad>=self.early_patience:self._early_stop=True; state['stop']=True
  (Path(self.output_folder)/'r260_early_stop.json').write_text(json.dumps(state,indent=2)); self.print_to_log_file('R260 early stop',json.dumps(state))
 def run_training(self):
  self.on_train_start()
  while self.current_epoch<self.num_epochs and not self._early_stop:
   self.on_epoch_start(); self.on_train_epoch_start(); train=[self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)]; self.on_train_epoch_end(train)
   with torch.no_grad():
    self.on_validation_epoch_start(); val=[self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)]; self.on_validation_epoch_end(val)
   self.on_epoch_end()
  self.on_train_end()
