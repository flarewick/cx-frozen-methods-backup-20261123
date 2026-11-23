"""Actual entry dispatch and real arrived-readout class integration on CPU."""
from pathlib import Path
from types import SimpleNamespace
import json,numpy as np,torch
from . import episode
from native_feedback_application_v1 import episode as actual, capture,readouts
from native_best100_endpoint_v1 import capture as old_capture
from native_feedback_application_v1.completion import CompletionQueue
from .contracts import check

def hook_test(cls,kind):
 recorder=SimpleNamespace(rows=[],dt=[],failed=False);seen=[];calls=[]
 def reader(t,s):
  calls.append(t);return dict(rgb=(np.zeros((240,320,3),np.uint8),)*3,command_state=np.zeros(14),images={'0':'same_actual_fixture'},physical_sha256='fixed_fixture',rng_sha256='fixed_fixture')
 initial=reader(0,0);initial['command_state'][[6,13]]=1;base=np.zeros((32,14),np.float32)
 hook=cls(recorder,reader,kind=kind,initial=initial,base=base,on_arrival=lambda t,s,obs,r:seen.append(t))
 for i in range(1,33):
  before=len(recorder.rows);recorder.rows.append(dict(gripper_logical_command=np.array([0. if i>=10 else 1.,0. if i>=20 else 1.])));recorder.dt.append(.004);hook.after(before)
 hook.terminal_read(32,complete_native_chunk=True);hook.completion.finish_origin('origin',arrived_tick=32,commands_returned=32,native_censored=False);hook.completion.observe_rgb('origin',observation_tick=32,decision_tick=32,frame_id='fixture_end')
 assert seen==[r['tick']for r in hook.readouts]and len(seen)==len(set(seen))
 return hook,calls,seen

def main():
 torch.set_num_threads(2);check();assert episode.derived_execute is actual.execute and capture.CurrentReadouts is readouts.CurrentReadouts and capture.CurrentReadouts is not old_capture.CurrentReadouts
 joint,_,arrived=hook_test(readouts.CurrentReadouts,('close_left','close_right'));assert isinstance(joint.completion,CompletionQueue)and joint.completion.closed[0]['requested95_tick']==20 and 10 in arrived and 20 in arrived
 a,ca,aa=hook_test(readouts.CurrentReadouts,'close_left');b,cb,ab=hook_test(old_capture.CurrentReadouts,'close_left');assert ca==cb and aa==ab and a.completion.closed==b.completion.closed
 T=Path(__file__).resolve().parents[1];O=T/'outputs/native_object_transfer_pilot_v3/development';O.mkdir(parents=True,exist_ok=True);value=dict(passed=True,actual_episode_dispatches_new_application=True,new_capture_uses_joint_readouts=True,joint_arrivals_once_and_actual_maximum_phase=True,single_kind_arrivals_and_receipts_exact_old=True,CUDA_initialized=torch.cuda.is_initialized(),no_native_rollout_claim=True)
 with(O/'CPU_integration_checks.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 print(json.dumps(value))
if __name__=='__main__':main()
