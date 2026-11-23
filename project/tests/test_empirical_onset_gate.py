import unittest
import numpy as np
from cx_tta.empirical_onset_gate import EmpiricalOnsetGate,candidate_packet,selected_actions
from cx_tta.arm_prescription_policy import compose

class Model:
    def __init__(self,rescue,harm):self.calls=0;self.rescue=rescue;self.harm=harm
    def predict(self,p):
        self.calls+=1
        return {'class_scores':dict(both_fail=0.,rescue=self.rescue,harm=self.harm,both_success=1-self.rescue-self.harm)}

class EmpiricalGateTests(unittest.TestCase):
    def packet(self):
        shapes=dict(visual=(3,2,2,32),text=(2,32),state=(14,),ee_pose=(2,9),past_visual=(3,2,2,32),past_text=(2,32),past_state=(14,),past_ee_pose=(2,9),has_history=(1,))
        legal={k:np.zeros((1,*v),np.float32) for k,v in shapes.items()}
        legal['base']=np.zeros((1,32,14),np.float32)
        return candidate_packet(legal,np.full((32,14),.03,np.float32))
    def test_rejection_consumes_opportunity_and_never_retries(self):
        model=Model(.1,.2);g=EmpiricalOnsetGate(model);g.step(0,[1,3])
        self.assertFalse(g.step(1,[1,3],self.packet())['accepted'])
        for q in (2,3,4):self.assertFalse(g.step(q,[1,3])['accepted'])
        self.assertEqual(model.calls,1)
        with self.assertRaises(ValueError):g.step(5,[6,8])
    def test_accept_one_chunk_reset_and_exact_actions(self):
        model=Model(.7,.1);g=EmpiricalOnsetGate(model);g.step(0,[1,3]);p=self.packet()
        decision=g.step(1,[1,3],p);self.assertTrue(decision['accepted'])
        base=p['base'][0];raw=p['candidate_delta'][0]
        a,d,_=selected_actions(base,raw,1,[1,3],decision)
        expected=compose(base,raw,'F_first_full',1,[1,3])
        self.assertEqual(a.tobytes(),expected[0].tobytes());self.assertEqual(d.tobytes(),expected[1].tobytes())
        a,d,_=selected_actions(base,raw,2,[1,3],g.step(2,[1,3]))
        self.assertEqual(a.tobytes(),base.tobytes());self.assertFalse(d.any())
        g.reset();self.assertFalse(g.step(0,None)['consumed'])
    def test_tie_rejects_and_no_lookahead_packet(self):
        g=EmpiricalOnsetGate(Model(.5,.25));g.step(0,[1,3])
        self.assertFalse(g.step(1,[1,3],self.packet())['accepted'])
        with self.assertRaises(ValueError):g.step(2,[1,3],self.packet())
        with self.assertRaises(ValueError):EmpiricalOnsetGate(Model(.7,.1)).step(1,None)
    def test_rejects_privileged_fields(self):
        g=EmpiricalOnsetGate(Model(.7,.1));g.step(0,[1,3])
        with self.assertRaises(ValueError):g.step(1,[1,3],dict(self.packet(),task='cup'))

if __name__=='__main__':unittest.main()
