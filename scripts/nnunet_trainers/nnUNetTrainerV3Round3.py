"""TSRS A/B joint heatmap input with frozen Round3 200-update budget."""
import copy,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from structural_prior_v3 import build_backbone_input
from local_supervision_v3 import local_supervision
from r349_parameter_gradient_gate_v3 import route_parameter_gradients_v3,assign_gradients_v3
from round3_common import sha,save,load_lock,require_manifest,resource_guard


class JointInputNetwork(nn.Module):
    def __init__(self,backbone,arm):
        super().__init__();self.backbone=backbone;self.arm=arm
    @property
    def decoder(self):return self.backbone.decoder
    def forward(self,data):
        return self.backbone(build_backbone_input(data[:,:1],data[:,1:2] if self.arm=='B' else None,self.arm))


class nnUNetTrainerV3Round3(nnUNetTrainer):
    def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.arm=os.environ['V3_ARM']
        if self.arm not in ('A','B'):raise ValueError('TSRS C/D supervision pending')
        self.protocol=Path(os.environ['V3_PROTOCOL']);self.lock=load_lock(self.protocol);require_manifest(self.protocol)
        cfg=self.lock['pilot'];self.cfg=cfg
        self.initial_lr=cfg['backbone_lr'];self.num_epochs=cfg['max_epochs'];self.num_iterations_per_epoch=cfg['updates_per_epoch']
        self.num_val_iterations_per_epoch=cfg['TSRS_val_iterations'];self.teacher=None;self.started=time.time()
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])

    def _set_batch_size_and_oversample(self):
        super()._set_batch_size_and_oversample();self.batch_size=1

    @staticmethod
    def build_network_architecture(plans_manager,configuration_manager,num_input_channels,num_output_channels,enable_deep_supervision=True):
        base=nnUNetTrainer.build_network_architecture(plans_manager,configuration_manager,num_input_channels,num_output_channels,enable_deep_supervision)
        return JointInputNetwork(base,os.environ['V3_ARM'])

    def initialize(self):
        if self.was_initialized:return
        super().initialize()
        path=Path(os.environ['V3_NATIVE']);expected=self.lock['native_sha256']['TSRS']
        if sha(path)!=expected:raise ValueError('native SHA mismatch')
        state=torch.load(path,map_location='cpu',weights_only=False)['network_weights']
        self.network.backbone.load_state_dict(state,strict=True)
        self.teacher=copy.deepcopy(self.network.backbone).eval().requires_grad_(False)
        self.manifest={'status':'RUNNING','dataset':'TSRS','arm':self.arm,'source_checkpoint':str(path),'source_sha256':expected,
                       'protocol_sha256':sha(self.protocol),'config':self.cfg,'seed':self.cfg['seed'],'input_contract':'image+seam B; image+zero A',
                       'code_sha256':sha(Path(__file__)),'test_used':False,'source_epoch':228,'selection':'EMA foreground Dice',
                       'split_sha256':sha(Path(self.preprocessed_dataset_folder_base)/'splits_final.json')}
        target=Path(self.output_folder)/'run_manifest.json'
        if target.exists():raise FileExistsError(target)
        save(target,self.manifest)

    def configure_optimizers(self):
        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        opt=torch.optim.AdamW(self.network.parameters(),lr=self.initial_lr,weight_decay=1e-4)
        return opt,PolyLRScheduler(opt,self.initial_lr,self.num_epochs)

    def train_step(self,batch):
        resource_guard(self.output_folder,self.started)
        data=batch['data'].to(self.device);target=batch['target']
        target=[t.to(self.device) for t in target] if isinstance(target,(list,tuple)) else target.to(self.device)
        y=target[0] if isinstance(target,list) else target
        self.optimizer.zero_grad(set_to_none=True);self.teacher.eval()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.network(data);base=self.loss(output,target)
            if self.arm=='B':
                with torch.no_grad():
                    ref=self.teacher(torch.cat((data[:,:1],torch.zeros_like(data[:,:1])),1));ref=ref[0] if isinstance(ref,(list,tuple)) else ref
                    teacher=(ref[:,1:2].float()-ref[:,:1].float()).sigmoid()
        rows=[];extra={}
        if self.arm=='A':base.backward()
        else:
            high=output[0] if isinstance(output,(list,tuple)) else output
            margin=high[:,1:2].float()-high[:,:1].float();valid=y>=0;truth=(y>0).float()
            terms,states=local_supervision(margin,truth,torch.zeros_like(truth,dtype=torch.bool),truth.bool(),valid,valid,data[:,1:2],teacher,
                                          enable_seam=True,enable_overlap=False)
            aux=[(k,terms[k],self.cfg[k+'_weight']*states['attenuation'][k]) for k in ['separation','preserve']]
            gradients,rows=route_parameter_gradients_v3(self.network.parameters(),base,aux,max_aux_ratio=.15,max_merged_aux_ratio=.15)
            assign_gradients_v3(self.network.parameters(),gradients)
            extra={'state_mass':states['masses'],'losses':{k:float(v.detach()) for k,v in terms.items()}}
        if not torch.isfinite(base):raise ValueError('nonfinite base loss')
        norm=torch.nn.utils.clip_grad_norm_(self.network.parameters(),self.cfg['gradient_clip_norm'],error_if_nonfinite=True)
        self.optimizer.step()
        with (Path(self.output_folder)/'routing.jsonl').open('a') as f:
            f.write(json.dumps({'epoch':self.current_epoch+1,'base_loss':float(base.detach()),'grad_norm':float(norm),'routing':rows,**extra})+'\n')
        return {'loss':base.detach().float().cpu().numpy()}

    def save_checkpoint(self,filename):
        # Keep one selected checkpoint; no final/latest duplicate or optimizer archive.
        if Path(filename).name!='checkpoint_best.pth':return
        super().save_checkpoint(filename)
        p=torch.load(filename,map_location='cpu',weights_only=False)
        p['optimizer_state']=None;p['grad_scaler_state']=None
        p['v3_arm']=self.arm;p['v3_protocol_sha256']=sha(self.protocol)
        torch.save(p,filename)

    def on_train_end(self):
        super().on_train_end()
        path=Path(self.output_folder)/'checkpoint_best.pth'
        payload=torch.load(path,map_location='cpu',weights_only=False)
        self.manifest.update({'status':'COMPLETE','selected_checkpoint':str(path),'selected_checkpoint_sha256':sha(path),
                              'selected_epoch':payload['current_epoch'],'updates':self.num_epochs*self.num_iterations_per_epoch,
                              'seconds':time.time()-self.started,'max_gpu_bytes':torch.cuda.max_memory_allocated()})
        save(Path(self.output_folder)/'run_manifest.json',self.manifest)

    def perform_actual_validation(self,*args,**kwargs):
        self.print_to_log_file('Round3 selected-checkpoint validation is performed by explicit frozen predictor after training')
