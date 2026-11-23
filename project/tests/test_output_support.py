import unittest
import numpy as np
import torch
from cx_tta.output_support_adapter import OutputSupportAdapter,CONDITIONS,support_loss,validate_native_arms
from cx_tta.structured_action_adapter import StructuredActionAdapter,action_direction_loss


def inputs():
    torch.manual_seed(92)
    return dict(visual=torch.randn(2,3,2,2,4),text=torch.randn(2,2,4),state=torch.randn(2,14),
        past_visual=torch.randn(2,3,2,2,4),past_text=torch.randn(2,2,4),past_state=torch.randn(2,14),
        ee_pose=torch.randn(2,2,9),past_ee_pose=torch.randn(2,2,9),has_history=torch.ones(2,1))


class OutputSupportContracts(unittest.TestCase):
    def test_identical_allocation_zero_and_old_bounded_values(self):
        old=StructuredActionAdapter(4,np.zeros(24));state=old.state_dict();x=inputs();base=torch.full((2,32,14),-0.)
        for condition in CONDITIONS:
            model=OutputSupportAdapter(4,np.zeros(24),condition);model.load_state_dict(state)
            self.assertEqual(set(state),set(model.state_dict()))
            self.assertEqual(sum(p.numel() for p in old.parameters()),sum(p.numel() for p in model.parameters()))
            self.assertTrue(torch.equal(model(x,base)['actions'],base))
            self.assertFalse(model(x,base)['delta'].detach().any())
        with torch.no_grad():old.action_head[-1].bias.fill_(2.)
        for condition in ('A_bounded_clipped','C_bounded_raw'):
            model=OutputSupportAdapter(4,np.zeros(24),condition);model.load_state_dict(old.state_dict())
            for k,v in old(x,base).items():self.assertTrue(torch.equal(v,model(x,base)[k]))

    def test_linear_support_and_bypass_preserve_signed_zero(self):
        model=OutputSupportAdapter(4,np.zeros(24),'B_open_clipped');x=inputs();base=torch.full((2,32,14),-0.)
        with torch.no_grad():model.action_head[-1].bias.fill_(3.)
        out=model(x,base);self.assertTrue(torch.allclose(out['delta'],model.caps.expand_as(base)*3))
        self.assertEqual(model(x,base,False)['actions'].detach().numpy().tobytes(),base.numpy().tobytes())

    def test_supervision_transforms_and_mask_gradients(self):
        scale=torch.tensor([.1]*6+[.2]+[.1]*6+[.2]);base=torch.zeros(2,32,14)
        pred=torch.full_like(base,.03,requires_grad=True);target=base+scale*4
        valid=torch.ones(2,32,dtype=torch.bool);valid[:,29:]=False;weights=torch.tensor([1.,0.])
        old=action_direction_loss(pred,base,target,valid,weights,scale)
        self.assertTrue(torch.equal(old,support_loss(pred,base,target,valid,weights,scale,True)))
        for clip in (True,False):
            loss=support_loss(pred,base,target,valid,weights,scale,clip);g=torch.autograd.grad(loss,pred)[0]
            changed=target.clone();changed[~valid]=float('nan');changed[1]=float('nan')
            loss2=support_loss(pred,base,changed,valid,weights,scale,clip);g2=torch.autograd.grad(loss2,pred)[0]
            self.assertTrue(torch.equal(loss,loss2));self.assertTrue(torch.equal(g,g2))
        self.assertGreater(float(support_loss(pred,base,target,valid,weights,scale,False)),float(old))

    def test_robot_guard_preserves_native_gripper_path(self):
        limits=np.array([[-10,10]]*14,np.float32);limits[[6,13]]=[0,1]
        a=np.zeros((32,14),np.float32);a[:,6]=-.01
        before=a.tobytes();stats=validate_native_arms(a,limits)
        self.assertEqual(stats['nominal_gripper_outside_0_1_coordinates'],32);self.assertEqual(a.tobytes(),before)
        a[0,0]=11
        with self.assertRaises(ValueError):validate_native_arms(a,limits)


if __name__=='__main__':unittest.main()
