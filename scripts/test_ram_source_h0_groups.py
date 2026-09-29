import unittest
import numpy as np
from analyze_ram_source_h0_groups import analyze


class H0GroupsTest(unittest.TestCase):
    def test_member_and_unique_pixel_cases(self):
        gt=np.zeros((14,1,4),bool);pred=np.zeros_like(gt);h0=np.zeros_like(gt)
        gt[0,0]=True;gt[1,0]=True
        h0[0,0,0]=True;h0[1,0,0]=True
        pred[0,0,1]=True;h0[1,0,1]=True
        h0[1,0,2]=True
        pred[0,0,3]=True;pred[2,0,3]=True;h0[1,0,3]=True
        row=analyze(gt,np.ones((1,4),bool),pred,h0)
        self.assertEqual(row['H0_member_pixels'],5)
        self.assertEqual(row['H0_unique_pixels'],4)
        self.assertEqual(row['H0_pred_members_0'],2)
        self.assertEqual(row['H0_pred_members_1'],1)
        self.assertEqual(row['H0_pred_members_2plus'],1)
        self.assertEqual(row['global_overlap_FN_recovered_by_oracle'],2)
        self.assertEqual(row['one_H0_member_sufficient'],1)
        self.assertEqual(row['at_least_two_H0_members_needed_and_available'],1)
        self.assertEqual(row['H0_alone_insufficient_from_pred0'],1)


if __name__=='__main__':unittest.main()
