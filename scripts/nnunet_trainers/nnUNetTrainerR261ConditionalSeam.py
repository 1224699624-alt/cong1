"""Mature R202 initialized nnU-Net with per-slot seam/support relation loss."""
from __future__ import annotations
import hashlib,json,math,os
from contextlib import nullcontext
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

class nnUNetTrainerR261ConditionalSeam(nnUNetTrainer):
 expected_dataset='Dataset205_TSRS_RSNAEpiphysisConditionalSeam2D'
 checkpoint_env='R261_ADAPTED_CHECKPOINT'
 run_prefix='r261'
 slot_weights=(1.,1.,1.,1.,1.,1.)
 slot_margins=(.15,.15,.15,.15,.15,.15)
 slot_side_floors=(.70,.70,.70,.70,.70,.70)
 def __init__(self,plans:dict,configuration:str,fold:int,dataset_json:dict,device:torch.device=torch.device('cuda')):
  super().__init__(plans,configuration,fold,dataset_json,device)
  if self.plans_manager.dataset_name!=self.expected_dataset: raise RuntimeError(self.plans_manager.dataset_name)
  self.initial_lr=1e-4; self.num_epochs=60; self.num_iterations_per_epoch=250; self.num_val_iterations_per_epoch=50
  self.relation_margin=.15; self.side_floor=.70; self.side_keep_weight=.50; self.gradient_ratio=.05; self.alpha_max=.12; self.warmup_epochs=5
  self.early_min_epochs=15; self.early_patience=10; self.early_min_delta=1e-4; self._early_best=None; self._early_bad=0; self._early_stop=False
  np.random.seed(261); torch.manual_seed(261); torch.cuda.manual_seed_all(261)

 def initialize(self):
  if self.was_initialized:return
  if os.environ.get('nnUNet_compile','').lower()!='false': raise RuntimeError('R261 requires nnUNet_compile=false')
  super().initialize(); path=Path(os.environ[self.checkpoint_env]); payload=torch.load(path,map_location=self.device,weights_only=False); state=payload['network_weights']; self.network.load_state_dict(state,strict=True)
  first=state['encoder.stages.0.0.convs.0.conv.weight']
  if tuple(first.shape)!=(32,13,3,3) or int(torch.count_nonzero(first[:,1:])): raise RuntimeError('invalid R261 first convolution')
  heads=[k for k in state if 'seg_layers' in k]
  if not heads: raise RuntimeError('segmentation heads missing')
  manifest={'checkpoint':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'strict_all_network_keys_loaded':True,'input_channels':13,'new_channel_nonzero':0,'segmentation_head_tensors_loaded':len(heads)}
  (Path(self.output_folder)/f'{self.run_prefix}_initialization.json').write_text(json.dumps(manifest,indent=2)); self.print_to_log_file(f'{self.run_prefix.upper()} initialization',json.dumps(manifest))

 def configure_optimizers(self):
  opt=torch.optim.SGD(self.network.parameters(),self.initial_lr,weight_decay=self.weight_decay,momentum=.99,nesterov=True); return opt,PolyLRScheduler(opt,self.initial_lr,self.num_epochs)

 def _relation_loss(self,logits,target,data):
  probability=torch.softmax(logits.float(),dim=1)[:,1]; y=(target[:,0]>0).float(); priors=torch.clamp(data[:,1:].float()/255.,0,1)
  losses=[]; gap_values=[]; support_values=[]; confidences=[]; valid_per_sample=torch.zeros(len(data),device=logits.device)
  for b in range(len(data)):
   for slot in range(6):
    seam=priors[b,2*slot]; support=priors[b,2*slot+1]; confidence=torch.minimum(seam.max(),support.max())
    gap_w=seam*(1-y[b]); support_w=support*y[b]; gap_mass=gap_w.sum(); support_mass=support_w.sum()
    if float(confidence.detach())<.02 or float(gap_mass.detach())<2.0 or float(support_mass.detach())<4.0: continue
    pg=(gap_w*probability[b]).sum()/(gap_mass+1e-6); ps=(support_w*probability[b]).sum()/(support_mass+1e-6)
    relation=F.relu(self.slot_margins[slot]+pg-ps); keep=F.relu(self.slot_side_floors[slot]-ps); losses.append(self.slot_weights[slot]*confidence*(relation+self.side_keep_weight*keep)); gap_values.append(pg); support_values.append(ps); confidences.append(confidence); valid_per_sample[b]+=1
  if losses: value=torch.stack(losses).mean()
  else: value=logits.sum()*0
  stats={'valid_slots':valid_per_sample.mean(),'abstention_rate':(valid_per_sample==0).float().mean(),'gap_probability':torch.stack(gap_values).mean() if gap_values else value.detach()*0,'support_probability':torch.stack(support_values).mean() if support_values else value.detach()*0,'slot_confidence':torch.stack(confidences).mean() if confidences else value.detach()*0}
  return value,stats

 def train_step(self,batch):
  data=batch['data'].to(self.device,non_blocking=True); target=batch['target']; target=[x.to(self.device,non_blocking=True) for x in target] if isinstance(target,list) else target.to(self.device,non_blocking=True); high=target[0] if isinstance(target,list) else target; self.optimizer.zero_grad(set_to_none=True); context=autocast(self.device.type,enabled=True) if self.device.type=='cuda' else nullcontext()
  with context:
   output=self.network(data); base=self.loss(output,target); logits=output[0] if isinstance(output,(list,tuple)) else output; relation,stats=self._relation_loss(logits,high,data)
   if relation.requires_grad and float(relation.detach())>0:
    gb_t=torch.autograd.grad(base,logits,retain_graph=True,allow_unused=True)[0]; gr_t=torch.autograd.grad(relation,logits,retain_graph=True,allow_unused=True)[0]; gb=torch.linalg.vector_norm(gb_t.float()) if gb_t is not None else torch.zeros((),device=logits.device); gr=torch.linalg.vector_norm(gr_t.float()) if gr_t is not None else torch.zeros((),device=logits.device); cosine=torch.sum(gb_t.float()*gr_t.float())/(gb*gr+1e-8) if gb_t is not None and gr_t is not None else torch.zeros((),device=logits.device); warmup=float(np.clip((self.current_epoch+1)/self.warmup_epochs,0,1)); alpha=torch.clamp(self.gradient_ratio*gb.detach()/(gr.detach()+1e-8),0,self.alpha_max)*warmup
   else: gb=gr=cosine=alpha=torch.zeros((),device=logits.device)
   loss=base+alpha*relation
  if self.grad_scaler is not None:self.grad_scaler.scale(loss).backward(); self.grad_scaler.unscale_(self.optimizer)
  else:loss.backward()
  clip=float(torch.nn.utils.clip_grad_norm_(self.network.parameters(),12).detach().cpu())
  if self.grad_scaler is not None:self.grad_scaler.step(self.optimizer); self.grad_scaler.update()
  else:self.optimizer.step()
  p=torch.softmax(logits.float(),dim=1)[:,1]; core=1-F.max_pool2d(1-(high[:,0]>0).float(),5,1,2); core_p=(p*core).sum()/(core.sum()+1e-6)
  result={'loss':loss.detach().cpu().numpy(),'base_loss':base.detach().cpu().numpy(),'relation_loss':relation.detach().cpu().numpy(),'alpha':alpha.detach().cpu().numpy(),'g_base':gb.detach().cpu().numpy(),'g_relation':gr.detach().cpu().numpy(),'gradient_cosine':cosine.detach().cpu().numpy(),'core_probability':core_p.detach().cpu().numpy(),'clip_norm':np.asarray(clip)}; result.update({k:v.detach().cpu().numpy() for k,v in stats.items()}); return result

 def on_train_epoch_end(self,outputs):
  super().on_train_epoch_end(outputs); row={k:float(np.mean([float(np.asarray(x[k])) for x in outputs])) for k in outputs[0] if k!='loss'}; row['epoch']=int(self.current_epoch); row['lr']=float(self.optimizer.param_groups[0]['lr']); (Path(self.output_folder)/f'{self.run_prefix}_training_dynamics.jsonl').open('a',encoding='utf-8').write(json.dumps(row)+'\n'); self.print_to_log_file(f'{self.run_prefix.upper()} dynamics',json.dumps(row))

 def on_epoch_end(self):
  super().on_epoch_end(); score=float(self.logger.get_value('ema_fg_dice',step=-1)); improved=self._early_best is None or score>self._early_best+self.early_min_delta
  if improved:self._early_best=score; self._early_bad=0
  else:self._early_bad+=1
  state={'completed_epochs':int(self.current_epoch),'ema_dice':score,'early_best':self._early_best,'bad_epochs':self._early_bad,'stop':False}
  if self.current_epoch>=self.early_min_epochs and self._early_bad>=self.early_patience:self._early_stop=True; state['stop']=True
  (Path(self.output_folder)/f'{self.run_prefix}_early_stop.json').write_text(json.dumps(state,indent=2)); self.print_to_log_file(f'{self.run_prefix.upper()} early stop',json.dumps(state))

 def run_training(self):
  self.on_train_start()
  while self.current_epoch<self.num_epochs and not self._early_stop:
   self.on_epoch_start(); self.on_train_epoch_start(); train=[self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)]; self.on_train_epoch_end(train)
   with torch.no_grad(): self.on_validation_epoch_start(); val=[self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)]; self.on_validation_epoch_end(val)
   self.on_epoch_end()
  self.on_train_end()

class nnUNetTrainerR261ConditionalSeamSanity(nnUNetTrainerR261ConditionalSeam):
 def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
  super().__init__(plans,configuration,fold,dataset_json,device); self.num_epochs=2; self.num_iterations_per_epoch=10; self.num_val_iterations_per_epoch=5; self.early_min_epochs=99
