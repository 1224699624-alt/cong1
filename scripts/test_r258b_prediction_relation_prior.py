#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from train_r258b_prediction_relation_prior import RelationPriorNet,assign_proposals,pair_geometry,strict

p=[{"x":10.,"y":10.,"scale":5.},{"x":20.,"y":10.,"scale":5.}]
r=[{"cx":10.,"cy":10.,"area":25.},{"cx":20.,"cy":10.,"area":25.}]
assert assign_proposals(p,r,.6)==[0,1]
assert np.allclose(pair_geometry(p[0],p[1]),(2.,0.))
model=RelationPriorNet(); image=np.zeros((2,3,128,128),np.float32)
import torch
logit,heat=model(torch.from_numpy(image),torch.zeros(2,2)); assert logit.shape==(2,) and heat.shape==(2,1,128,128)
assert strict(float("inf")) is None
print("R258B synthetic checks: PASS")
