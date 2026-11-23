import unittest
import numpy as np
from cx_tta.onset_observables import at_query, validate_packet


class CausalOnsetPacket(unittest.TestCase):
    def setUp(self):
        shapes = dict(visual=(3,2,2,32), text=(2,32), state=(14,),
            past_visual=(3,2,2,32), past_text=(2,32), past_state=(14,),
            has_history=(1,), ee_pose=(2,9), past_ee_pose=(2,9))
        self.x = {k: np.zeros((4,*v), np.float32) for k,v in shapes.items()}
        self.x['has_history'][1:] = 1
        self.base = np.zeros((4,32,14), np.float32)
        self.raw = np.full_like(self.base, .04)

    def test_future_independence_and_copy(self):
        before = at_query(self.x, self.base, self.raw, 1)
        for value in (*self.x.values(), self.base, self.raw): value[2:] = np.nan
        after = at_query(self.x, self.base, self.raw, 1)
        self.assertTrue(all(before[k].tobytes()==after[k].tobytes() for k in before))
        after['state'].fill(9)
        self.assertTrue(np.all(self.x['state'][1]==0))

    def test_disabled_grip_and_privileged_fields(self):
        self.raw[:,:,6] = .2
        self.raw[:,:,13] = -.2
        p = at_query(self.x,self.base,self.raw,1)
        self.assertTrue(np.all(p['candidate_delta'][:,:,[6,13]]==0))
        for key in ('task','seed','success','GT_pose','future_state'):
            with self.assertRaises(ValueError): validate_packet(dict(p, **{key:0}))

    def test_native_bounds_and_invalid_query(self):
        self.raw[1,0,0] = .11
        with self.assertRaises(ValueError): at_query(self.x,self.base,self.raw,1)
        with self.assertRaises(ValueError): at_query(self.x,self.base,self.raw,-1)
        with self.assertRaises(ValueError): at_query(self.x,self.base,self.raw,4)


if __name__=='__main__': unittest.main()
