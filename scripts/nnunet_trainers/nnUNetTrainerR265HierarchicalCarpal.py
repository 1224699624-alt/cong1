"""Hierarchical carpal relation trainer initialized from mature R202."""
import torch
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerR261ConditionalSeam import nnUNetTrainerR261ConditionalSeam

class nnUNetTrainerR265HierarchicalCarpal(nnUNetTrainerR261ConditionalSeam):
 expected_dataset='Dataset206_TSRS_RSNAEpiphysisHierarchicalCarpal2D'
 checkpoint_env='R265_ADAPTED_CHECKPOINT'
 run_prefix='r265'
 slot_weights=(1.,1.,1.,1.,.45,.15)
 slot_margins=(.18,.18,.18,.18,.10,.05)
 slot_side_floors=(.72,.72,.72,.72,.68,.65)
 def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
  super().__init__(plans,configuration,fold,dataset_json,device)
  self.gradient_ratio=.04; self.alpha_max=.10; self.warmup_epochs=5

class nnUNetTrainerR265HierarchicalCarpalSanity(nnUNetTrainerR265HierarchicalCarpal):
 def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
  super().__init__(plans,configuration,fold,dataset_json,device); self.num_epochs=2; self.num_iterations_per_epoch=10; self.num_val_iterations_per_epoch=5; self.early_min_epochs=99
