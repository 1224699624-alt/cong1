"""TSRS pilot: frozen nnU-Net with input and output structural adapters."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from r351_dual_prior import ImageOverlapPrior, SeamInputAdapter, OutputCompletionPrior, local_terms, normalize_prior, gate_positive_completion
from r349_parameter_gradient_gate import route_parameter_gradients, assign_gradients


class DualPriorNetwork(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.input_prior = SeamInputAdapter()
        self.output_prior = OutputCompletionPrior()
        self.overlap_prior = ImageOverlapPrior().requires_grad_(False)

    @property
    def decoder(self):
        return self.backbone.decoder

    def forward(self, data):
        image, seam = data[:, :1], data[:, 1:2]
        with torch.no_grad():
            overlap = torch.sigmoid(self.overlap_prior(image.float(), seam.float()))
        adapted = self.input_prior(image, seam, overlap)
        output = self.backbone(torch.cat((adapted, seam), 1))
        high = output[0] if isinstance(output, (list, tuple)) else output
        margin = high[:, 1:2].float() - high[:, :1].float()
        refined = self.output_prior.refine(image.float(), margin, seam.float(), overlap, steps=1)
        if os.environ.get('R351_POSITIVE_GATE', '0') == '1':
            refined = gate_positive_completion(margin, refined, seam.float(), overlap)
        result = torch.cat((high[:, :1], high[:, :1] + refined), 1)
        return [result, *output[1:]] if isinstance(output, (list, tuple)) else result


class nnUNetTrainerR351DualPrior(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.plans_manager.dataset_name != 'Dataset204_TSRS_RSNAEpiphysisMaturePrior2D':
            raise ValueError(self.plans_manager.dataset_name)
        self.initial_lr = float(os.environ.get('R351_TSRS_LR', '2e-5'))
        self.num_epochs = int(os.environ.get('R351_EPOCHS', '2'))
        self.num_iterations_per_epoch = int(os.environ.get('R351_ITERATIONS', '100'))
        self.num_val_iterations_per_epoch = 25
        self.audit_only = os.environ.get('R351_AUDIT_ONLY', 'false') == 'true'
        np.random.seed(3512); torch.manual_seed(3512); torch.cuda.manual_seed_all(3512)

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        base = nnUNetTrainer.build_network_architecture(plans_manager, configuration_manager,
                  num_input_channels, num_output_channels, enable_deep_supervision)
        return DualPriorNetwork(base)

    def _set_batch_size_and_oversample(self):
        super()._set_batch_size_and_oversample()
        self.batch_size = min(self.batch_size, 2)

    def initialize(self):
        if self.was_initialized: return
        super().initialize()
        checkpoint = Path(os.environ['R351_TSRS_INIT'])
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.network.backbone.load_state_dict(state['network_weights'], strict=True)
        overlap = torch.load(os.environ['R351_OVERLAP_CHECKPOINT'], map_location=self.device, weights_only=False)
        self.network.overlap_prior.load_state_dict(overlap['model'], strict=True)
        ram = torch.load(os.environ['R351_RAM_INIT'], map_location='cpu', weights_only=False)
        init = self.network.output_prior.load_mature(ram['refiner'], identity=True)
        (Path(self.output_folder)/'r351_initialization.json').write_text(json.dumps({
            'backbone':str(checkpoint),'output_prior_transfer':init,
            'overlap_annotation':'unknown; no TSRS overlap target fabricated',
            'frozen_backbone':True,'overlap_prior_source':'RAM image-conditioned true overlap supervision'},indent=2))

    def configure_optimizers(self):
        params = [p for p in self.network.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.initial_lr, weight_decay=1e-4)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def train_step(self, batch):
        data = batch['data'].to(self.device)
        target = batch['target'][0] if isinstance(batch['target'], (list, tuple)) else batch['target']
        target = target.to(self.device)
        valid = (target[:, :1] >= 0).float()
        observed = (target[:, :1] > 0).float()
        self.optimizer.zero_grad(set_to_none=True)
        self.network.overlap_prior.eval(); self.network.backbone.eval()
        with torch.no_grad():
            original = self.network.backbone(data)
            original = original[0] if isinstance(original, (list, tuple)) else original
            teacher = torch.softmax(original.float(), 1)[:, 1:2]
            overlap = torch.sigmoid(self.network.overlap_prior(data[:, :1], data[:, 1:2]))
        output = self.network(data)
        high = output[0] if isinstance(output, (list, tuple)) else output
        margin = high[:, 1:2] - high[:, :1]
        terms, states = local_terms(margin, observed, teacher, data[:, 1:2], overlap, valid, False)
        ramp = min(1., (self.current_epoch + 1) / 5.)
        gradients, routing = route_parameter_gradients(self.network.parameters(), terms['base'], [
            ('separation', terms['separation'], .035*ramp),
            ('overlap', terms['overlap'], .035*ramp),
            ('preserve', terms['preserve'], .10*ramp)], max_aux_ratio=.15)
        if self.audit_only:
            difference = float((torch.softmax(high.detach().float(), 1)[:, 1:2] - teacher).abs().max())
            report = {'passed':difference < 1e-6 and all(torch.isfinite(x).all() for x in gradients if x is not None),
                      'identity_probability_max_abs':difference,'states':states,
                      'input_gradient_norm':sum(float(g.norm()) for (name,p),g in zip([(n,p) for n,p in self.network.named_parameters() if p.requires_grad],gradients) if g is not None and name.startswith('input_prior.')),
                      'all_adapter_gradient_norm':sum(float(g.norm()) for g in gradients if g is not None),
                      'routing':[vars(x) for x in routing]}
            (Path(self.output_folder)/'smoke_audit.json').write_text(json.dumps(report,indent=2))
            if not report['passed']: raise RuntimeError(report)
        else:
            assign_gradients(self.network.parameters(), gradients)
            torch.nn.utils.clip_grad_norm_([p for p in self.network.parameters() if p.requires_grad], 5)
            self.optimizer.step()
        return {'loss':terms['base'].detach().cpu().numpy(),
                **{k:np.asarray(float(v.detach())) for k,v in terms.items()},
                **{k:np.asarray(v,dtype=np.float32) for k,v in states.items()}}

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)
        row={k:float(np.mean([float(x[k]) for x in outputs])) for k in outputs[0]}
        row['epoch']=int(self.current_epoch)+1
        with (Path(self.output_folder)/'r351_history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')

    def run_training(self):
        if not self.audit_only: return super().run_training()
        self.on_train_start(); self.on_epoch_start(); self.on_train_epoch_start()
        self.train_step(next(self.dataloader_train))
        self.print_to_log_file('R351 TSRS real batch audit completed')
