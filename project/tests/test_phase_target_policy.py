"""Meaningful composition contracts for canonical phase labels."""
import unittest
import numpy as np
from cx_tta.phase_target_policy import compose_grip,GRIPPERS,CASES
class PhaseComposition(unittest.TestCase):
    def test_zero_and_arms_identity(self):
        b=np.full((32,14),-.0,np.float32);r=np.full_like(b,.1)
        for c in CASES:
            goal=np.asarray([.8,.1],np.float32) if c=='T_stage_grip' else None
            a,d=compose_grip(b,r,c,False,goal)
            self.assertEqual(a.tobytes(),b.tobytes());self.assertFalse(d.any())
            a,d=compose_grip(b,r,c,True,goal)
            arm=[i for i in range(14) if i not in GRIPPERS]
            self.assertEqual(a[:,arm].tobytes(),b[:,arm].tobytes())
    def test_bound_and_dynamic_expert_hold_release(self):
        b=np.zeros((32,14),np.float32);b[:,6]=.619;b[:,13]=.09;r=np.zeros_like(b)
        a,d=compose_grip(b,r,'T_stage_grip',True,np.asarray([1.,0.],np.float32))
        np.testing.assert_array_equal(d[:,6],np.full(32,.2,np.float32));np.testing.assert_allclose(a[:,13],0.,atol=1e-8)
        self.assertTrue((a[:,6]>.8).all())
        b[:,6]=1.1;a,d=compose_grip(b,r,'C_base',True)
        np.testing.assert_array_equal(a,b) # no boundarypreclip
    def test_privilege_and_illegal_rejection(self):
        b=np.zeros((32,14),np.float32);r=np.zeros_like(b)
        for c in ('C_base','A_grip'):
            with self.assertRaises(ValueError):compose_grip(b,r,c,True,np.ones(2,np.float32))
        with self.assertRaises(ValueError):compose_grip(b,r,'T_stage_grip',True,None)
        with self.assertRaises(ValueError):compose_grip(b.astype(float),r,'C_base',True)
        r[0,6]=float('nan')
        with self.assertRaises(ValueError):compose_grip(b,r,'A_grip',True)
    def test_inputs_not_mutated_and_frozen_proposal_onlygrip(self):
        b=np.arange(448,dtype=np.float32).reshape(32,14)/1000;r=np.full_like(b,-.03);x=b.copy();y=r.copy()
        a,d=compose_grip(b,r,'A_grip',True)
        np.testing.assert_array_equal(d[:,GRIPPERS],r[:,GRIPPERS]);self.assertEqual(b.tobytes(),x.tobytes());self.assertEqual(r.tobytes(),y.tobytes())
if __name__=='__main__':unittest.main()

