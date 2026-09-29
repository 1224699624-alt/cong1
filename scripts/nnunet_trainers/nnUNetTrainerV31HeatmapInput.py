"""TSRS H_input: direct image+seam, native base loss only."""
import json,os
from pathlib import Path
import numpy as np
import torch
from .nnUNetTrainerV3Round3 import nnUNetTrainerV3Round3

class nnUNetTrainerV31HeatmapInput(nnUNetTrainerV3Round3):
    def __init__(self,*args,**kwargs):
        os.environ['V3_ARM']='B'  # build direct [image,seam] wrapper
        super().__init__(*args,**kwargs)
        self.arm='H_input'
        self.smoke_only=os.environ.get('V31_SMOKE_ONLY','false').lower()=='true'

    def initialize(self):
        super().initialize()
        v31=Path(os.environ['V31_PROTOCOL']);protocol=json.loads(v31.read_text())
        if protocol['status']!='LOCKED_BEFORE_UPDATE' or protocol['test_used']:raise ValueError('V3.1 protocol gate')
        from round3_common import sha
        if protocol['source_code_hashes']['nnunet_trainers/nnUNetTrainerV31HeatmapInput.py']!=sha(Path(__file__)):
            raise ValueError('H_input trainer differs from lock')
        self.manifest.update({'arm':'H_input','comparison_role':'pure joint heatmap input',
                              'auxiliary_losses':[],'gradient_routing':False,
                              'reference_arms':['Round3 A','Round3 B'],'v31_protocol_sha256':sha(v31),
                              'runner_sha256':sha(Path(__file__))})
        from round3_common import save
        save(Path(self.output_folder)/'run_manifest.json',self.manifest)

    def train_step(self,batch):
        from round3_common import resource_guard
        resource_guard(self.output_folder,self.started)
        data=batch['data'].to(self.device);target=batch['target']
        target=[t.to(self.device) for t in target] if isinstance(target,(list,tuple)) else target.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.network(data);base=self.loss(output,target)
        if not torch.isfinite(base):raise ValueError('nonfinite base loss')
        base.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.network.parameters(),self.cfg['gradient_clip_norm'],error_if_nonfinite=True)
        first=next(p for p in self.network.parameters() if p.ndim==4 and p.shape[1]==2)
        seam_grad=float(first.grad[:,1].float().norm())
        if not self.smoke_only:self.optimizer.step()
        with (Path(self.output_folder)/'routing.jsonl').open('a') as f:
            f.write(json.dumps({'epoch':self.current_epoch+1,'base_loss':float(base.detach()),
                                'grad_norm':float(norm),'seam_channel_gradient_norm':seam_grad,
                                'auxiliary_losses':[],'routing':'DISABLED_BASE_ONLY'})+'\n')
        return {'loss':base.detach().float().cpu().numpy()}

    def save_checkpoint(self,filename):
        super().save_checkpoint(filename)
        path=Path(filename)
        if path.name=='checkpoint_best.pth' and path.is_file():
            from round3_common import sha
            payload=torch.load(path,map_location='cpu',weights_only=False)
            payload['v31_protocol_sha256']=sha(Path(os.environ['V31_PROTOCOL']))
            payload['runner_sha256']=sha(Path(__file__))
            torch.save(payload,path)

    def run_training(self):
        if not self.smoke_only:return super().run_training()
        self.on_train_start();self.on_epoch_start();self.on_train_epoch_start()
        result=self.train_step(next(iter(self.dataloader_train)))
        record={'status':'PASS','optimizer_steps':0,'real_batch':True,'loss':float(result['loss']),
                'input_contract':'image+seam','auxiliary_losses':[],'routing':False,'test_used':False}
        (Path(self.output_folder)/'smoke.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
