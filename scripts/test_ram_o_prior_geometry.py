import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_ram_o_prior_geometry import load_h0, shifted


class RamOPriorGeometryTests(unittest.TestCase):
    def test_h0_little_endian_roundtrip_keeps_member_coordinates(self):
        original=np.zeros((14,2,3),dtype=bool)
        original[0,0,0]=True
        original[5,1,2]=True
        original[13,0,1]=True
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'h0.npz'
            np.savez_compressed(path,shape=np.asarray(original.shape),original_hw=np.asarray([2,3]),
                                h0=np.packbits(original.reshape(-1),bitorder='little'))
            decoded=load_h0(path,original.shape,(2,3))
        np.testing.assert_array_equal(decoded,original)

    def test_shift_diagnostic_zeros_exposed_border(self):
        original=np.arange(1,17,dtype=np.int32).reshape(4,4)
        expected=np.zeros_like(original)
        expected[1:,0:3]=original[0:3,1:]
        np.testing.assert_array_equal(shifted(original,1,-1),expected)
        np.testing.assert_array_equal(shifted(original,0,0),original)


if __name__=='__main__':unittest.main()
