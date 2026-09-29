import hashlib,tempfile,unittest
from pathlib import Path
import torch
from run_ram_source_correctability_2x2 import _load_step_payload


def digest(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


class SchedulePayloadTest(unittest.TestCase):
    def test_step_metadata_survives_payload_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'step.pt';image=torch.zeros((1,4,4),dtype=torch.float32)
            mask=torch.zeros((14,4,4),dtype=torch.float32);mask[1,0,0]=1
            valid=torch.ones((1,4,4),dtype=torch.bool);h0=torch.zeros((14,4,4),dtype=torch.bool);h0[1,0,0]=True
            values={'image':image,'mask':mask,'valid':valid,'h0':h0};torch.save(values,path)
            row={'path':path,'case':'unit_case','step':7,'indices':[0,1,2,3],'denominator_c0':64.,
                 'batch_sha256':hashlib.sha256(image.numpy().tobytes()+mask.numpy().tobytes()+valid.numpy().tobytes()).hexdigest(),
                 **{key+'_sha256':digest(value) for key,value in values.items()}}
            payload,_=_load_step_payload(row)
            self.assertEqual(payload['step'],7)
            self.assertEqual(payload['case'],'unit_case')
            self.assertEqual(int(payload['h0'].sum()),1)


if __name__=='__main__':unittest.main()
