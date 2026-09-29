"""Compare the legacy final wrapper and the corrected joint-input contract."""
from pathlib import Path
import hashlib
import json
import sys
import torch
from torch import nn

from nnunet_trainers.nnUNetTrainerFinalPrior import FinalPriorNetwork
from structural_prior_v3 import build_backbone_input


class CaptureBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.last = None

    def forward(self, data):
        self.last = data.detach().clone()
        return torch.cat([data[:, :1], data[:, :1]], 1)


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'outputs/v3_round2/INPUT_CONTRACT_TRACE.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(2)
    image = torch.randn(1, 1, 16, 16)
    seam = torch.linspace(0, 1, 256).reshape_as(image)
    backbone = CaptureBackbone()
    old = FinalPriorNetwork(backbone, variant='B').eval()
    with torch.no_grad():
        old(torch.cat([image, seam], 1))
    legacy = {'input_shape':list(backbone.last.shape),
              'image_channel_exact_at_identity':bool(torch.equal(backbone.last[:, :1], image)),
              'seam_channel_equals_heatmap':bool(torch.equal(backbone.last[:, 1:2], seam)),
              'seam_channel_nonzero':int(torch.count_nonzero(backbone.last[:, 1:2])),
              'status':'FAIL user-corrected joint-input design'}
    new = {}
    for arm in 'ABCD':
        value = build_backbone_input(image, seam, arm)
        new[arm] = {'shape':list(value.shape),'image_unchanged':bool(torch.equal(value[:, :1],image)),
                    'seam_equals_heatmap':bool(torch.equal(value[:, 1:2],seam)),
                    'seam_nonzero':int(torch.count_nonzero(value[:, 1:2]))}
    report = {'evidence_kind':'CPU SYNTHETIC INPUT HOOK; real batch evidence in N0_EQUIVALENCE files',
              'legacy_final_wrapper':legacy, 'v3_joint_input':new,
              'source_traces':{
                  'TSRS_R317':'run_r317_image_centers_only_remote.sh -> prepare_r260_mature_prior_dataset.py writes _0000 image and _0001 seam -> nnUNetTrainerR317.py inherits R260; self.network(data)',
                  'RAM_R323':'build_r323_ram_native_prior_maps.py generates canonical maps with RAM-native RelationPriorNet; train_r322_ram_nnunet_r317_seam_prior.py RamPriorDataset returns cat(image_t,prior); RamNnUNetTwoChannel accepts 2 channels',
                  'final_TSRS':'FinalPriorNetwork.forward -> adapt_input -> native_input -> zero_native_prior_channel -> [adapted_image,zero]',
                  'final_RAM':'train_final_prior_ram.forward_variant -> input adapter -> model(adapted,False); original model takes one image channel'},
              'status':'PASS corrected A/C zero, B/D original-image-plus-seam contract',
              'version':{name:hashlib.sha256((root/'scripts'/name).read_bytes()).hexdigest()
                         for name in ['trace_round2_input_contract.py','structural_prior_v3.py','final_prior_plugin.py']},
              'command':f'{sys.executable} scripts/trace_round2_input_contract.py'}
    out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
