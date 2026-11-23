"""CPU behavior contracts only, not native efficacy evidence."""
from pathlib import Path
from types import SimpleNamespace
import json,numpy as np,torch
from .application import FeedbackApplication,ARM
from .completion import CompletionQueue
from stage_feedback_v3.completion import CompletionQueue as Previous
from cx_tta.multi_event import EventConfig,LocalController

def event(q,active=False,adaptation=False):return dict(query=q,decision=dict(active=active,interventions=0,selected_event=None,phase='base'),adaptation_applied=adaptation)
def receipt(q,source=0,tag='one'):return dict(accepted=True,mode='TTA',query=q,source_query=source,fast_after=tag,returned_direction_changed=True)
def run_application():
 base=np.full((32,14),.2,np.float32);ref=np.zeros_like(base);ref[:,ARM]=.04;direction=ref[:,ARM].copy()+np.float32(.01);delta=np.zeros_like(base)
 c=SimpleNamespace(controller=LocalController(EventConfig()));c.controller.last_intervention=0;c.controller.interventions=1;app=FeedbackApplication()
 a,d,e,r=app.step(c,base,ref,direction,base.copy(),delta.copy(),event(1),receipt(1),condition='TTA');assert np.array_equal(a,base)and c.controller.interventions==1 and r['reason']=='original_base_only_cooldown'
 a,d,e,r=app.step(c,base,ref,direction,base.copy(),delta.copy(),event(2),dict(accepted=False),condition='TTA');assert r['applied']and c.controller.interventions==2 and np.array_equal(d[:,ARM],direction-ref[:,ARM])and np.all(d[:,[6,13]]==0)and np.array_equal(a[:,[6,13]],base[:,[6,13]])
 assert np.max(np.abs(d[:,ARM]))<.011,'Must not add frozen reference at extra query'
 c.controller.interventions=3;c.controller.last_intervention=4
 a,d,e,r=app.step(c,base,ref,direction,base.copy(),delta.copy(),event(6),receipt(6,4,'two'),condition='TTA');assert not r['applied']and np.array_equal(a,base)and c.controller.interventions==3
 try:app.step(c,base,ref,direction,base.copy(),delta.copy(),event(6),receipt(6,4,'two'),condition='TTA')
 except ValueError:pass
 else:raise AssertionError('Accepted revision reused')
 for condition in ('M_event_full','zero_update','shadow_update'):
  fresh=FeedbackApplication();a0=base.copy();d0=delta.copy();e0=event(2);a,d,e,r=fresh.step(SimpleNamespace(controller=LocalController(EventConfig())),base,ref,direction,a0,d0,e0,receipt(2),condition=condition);assert a is a0 and d is d0 and e is e0 and not fresh.pending and not fresh.seen
 c=SimpleNamespace(controller=LocalController(EventConfig()));fresh=FeedbackApplication()
 try:fresh.step(c,base,ref,ref[:,ARM]+np.float32(.03),base.copy(),delta.copy(),event(2),receipt(2),condition='TTA')
 except ValueError:pass
 else:raise AssertionError('Trust shift exceeded')
 assert c.controller.interventions==0
 c=SimpleNamespace(controller=LocalController(EventConfig()));c.controller.interventions=1;c.controller.last_intervention=2;fresh=FeedbackApplication();original=base+ref;a,d,e,r=fresh.step(c,base,ref,direction,original,ref.copy(),event(2,True,True),receipt(2),condition='TTA');assert a is original and r['applied']and c.controller.interventions==1 and fresh.pending is None
 c=SimpleNamespace(controller=LocalController(EventConfig()));fresh=FeedbackApplication();u=receipt(2);u['returned_direction_changed']=False
 a,d,e,r=fresh.step(c,base,ref,ref[:,ARM].copy(),base.copy(),delta.copy(),event(2),u,condition='TTA');assert not r['applied']and c.controller.interventions==0
 a,d,e,r=fresh.step(c,base,ref,direction,base.copy(),delta.copy(),event(3),dict(accepted=False),condition='TTA');assert r['applied']and c.controller.interventions==1
 return dict(accepted_no_current_visible_shift_deferred_not_error=True,fast_only_difference_not_frozen_readout=True,original_cooldown_preserved=True,total_three_budget_preserved=True,revision_single_use=True,full_zero_shadow_byte_exact=True,over_trust_shift_rejected=True,original_active_event_never_double_applied=True)
def complete(cls,kind,missing=False):
 c=cls();base=np.zeros((32,14),np.float32);c.register('event','origin',kind,np.ones(2),base,decision_tick=0,frame_id='actual_fixture0')
 for i in range(1,33):c.after_actuation('origin',np.array([0. if i>=10 else 1.,0. if i>=20 and not missing else 1.]),committed_tick=i,dt_s=.004)
 c.finish_origin('origin',arrived_tick=32,commands_returned=32,native_censored=False);c.observe_rgb('origin',observation_tick=32,decision_tick=32,frame_id='actual_fixture32');return c.closed

def main():
 torch.set_num_threads(2);app=run_application();joint=complete(CompletionQueue,('close_left','close_right'));assert joint[0]['requested95_tick']==20 and [x['requested95_tick']for x in joint[0]['components']]==[10,20]
 missing=complete(CompletionQueue,('close_left','close_right'),True);assert missing[0]['requested95_tick']is None and missing[0]['status']=='censored'
 assert complete(CompletionQueue,'close_left')==complete(Previous,'close_left')
 try:complete(CompletionQueue,('close_left','open_left'))
 except ValueError:pass
 else:raise AssertionError('Contradictory same arm admitted')
 T=Path(__file__).resolve().parents[1];O=T/'outputs/native_feedback_application_v1';O.mkdir(parents=True,exist_ok=True);p=O/'CPU_contract_checks_v2.json';value=dict(passed=True,application=app,joint_phase_latest_actual_side=True,missing_side_never_credited=True,single_kind_exact_previous=True,contradictory_side_rejected=True,CUDA_initialized=torch.cuda.is_initialized(),no_native_efficacy_claim=True)
 with p.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 print(json.dumps(value))
if __name__=='__main__':main()
