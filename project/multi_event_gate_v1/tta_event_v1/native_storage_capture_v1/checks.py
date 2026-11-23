import ast,copy,threading,uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped,verify
from .scope import restore,INSTALL,HOST
def startup_check():
 from native_onpolicy_arm_collection_v31.startup import StagedLanes,actor_class
 cancel=threading.Event();events=[];gate=StagedLanes(cancel,events.append,lanes=10)
 entered=[threading.Event()for _ in range(10)];admitted=[threading.Event()for _ in range(10)]
 def worker(n):
  entered[n].set()
  if gate.wait(n):admitted[n].set()
 threads=[threading.Thread(target=worker,args=(n,))for n in range(10)]
 for p in threads:p.start()
 try:
  assert all(e.wait(2)for e in entered)and all(e.wait(2)for e in admitted[:4])and not admitted[4].wait(.05)
  initial=dict(native_action_count=0,rgb=[np.zeros((240,320,3),np.uint8)for _ in range(3)],command_state=np.zeros(14))
  assert gate.release_q0(initial,'case0','0'*64)is False and not gate.eight.is_set()
  token=object();calls=[];marked=[]
  class Original:
   def query(self,*args):calls.append(args);return token
  actor=actor_class(Original,gate,{'case0':0},SimpleNamespace(mark_case_ready=marked.append))()
  folder=Path('/CPU_fixture/case0/M_event_full')
  assert actor.query(None,dict(query=1,global_tick=10),folder)is token and not gate.eight.is_set()
  assert actor.query(None,dict(query=0,global_tick=0),folder)is token and len(calls)==2 and marked==['case0']
  assert all(e.wait(2)for e in admitted)and len(events)==1 and events[0]['released_lanes']==list(range(4,10))
 finally:
  cancel.set()
  for p in threads:p.join(2);assert not p.is_alive()
 error=ValueError('original failure')
 class Failing:
  def query(self,*args):raise error
 failed=StagedLanes(threading.Event(),lambda x:None,lanes=10)
 actor=actor_class(Failing,failed,{'case0':0},None)()
 try:actor.query(None,dict(query=0,global_tick=0),folder)
 except ValueError as e:assert e is error
 else:raise AssertionError('Original query error suppressed')
 assert not failed.eight.is_set()
 return dict(passed=True,real_CPU_threads=10,only_first4_initially=True,passive_q0_cannot_release=True,
  actual_once_q0_releases6=True,original_call_return_exception_preserved=True,no_GPU_or_physics=True)
def main():
 import torch
 torch.set_num_threads(1);assert not torch.cuda.is_initialized()
 old=T/'native_onpolicy_arm_collection_v29';new=T/'native_onpolicy_arm_collection_v31';out=T/'outputs/native_onpolicy_arm_collection_v31';out.mkdir(parents=True,exist_ok=True)
 proof=T/'outputs/native_onpolicy_arm_collection_v29/CPU_contract_checks_v3.json';v=read(proof);assert v['passed'];verify(v)
 expected=lambda name:(old/name).read_text().replace(old.name,new.name)
 exceptions={'AGENTS.md','operational_checks.py','runtime.py','preload.py','host.py','startup.py','dispatch_checks.py','collection_plan.py'}
 for p in old.rglob('*'):
  if p.is_file()and p.suffix in('.py','.sh','.md'):
   q=new/p.relative_to(old)
   if q.suffix=='.py':ast.parse(q.read_text())
   if str(p.relative_to(old))not in exceptions:assert q.read_text()==expected(p.relative_to(old)),str(q)
 assert restore((new/'runtime.py').read_text())==expected('runtime.py')
 assert (new/'preload.py').read_text().replace(INSTALL,'')==expected('preload.py')
 host=(new/'host.py').read_text().replace(HOST,'').replace(",root/'CPU_storage_whole25_audit.json'",'')
 assert host==expected('host.py'),'Original resource/normal-exit guards changed'
 plan=(new/'collection_plan.py').read_text();start=plan.index("  front_file=T/");end=plan.index("  v['unread_capture_whole_reference_each_case']=True",start);plan=plan[:start]+plan[end:]
 plan=plan.replace('simulation_lanes=10,cases_per_batch=25','simulation_lanes=9,cases_per_batch=25').replace("v['startup_release_requires_actual_complete_initial_native32']=False","v['startup_release_requires_actual_complete_initial_native32']=True").replace('native_onpolicy_calibration_data_v26.reader','native_onpolicy_calibration_data_v25.reader')
 plan=plan.replace('outputs/native_onpolicy_arm_collection_v31/CPU_contract_checks_v1.json','outputs/native_onpolicy_arm_collection_v31/CPU_contract_checks_v3.json').replace('outputs/native_onpolicy_arm_collection_v31/CPU_actual_host_exit_v1.json','outputs/native_onpolicy_arm_collection_v31/CPU_actual_host_exit_v3.json').replace("all_failed_seconds_retained=v['inherited_stage_GPU_seconds']",'all_failed_seconds_retained=12409.593')
 assert plan==expected('collection_plan.py')
 startup=startup_check()
 from native_onpolicy_arm_collection_v31.dispatch_checks import run
 routing=run();assert routing['native_lanes_fixture']==10 and routing['passed']
 from .source import tracked
 fixture=out/'CPU_source_fixture'/uuid.uuid4().hex;fixture.mkdir(parents=True)
 token=object();calls=[];mask=np.zeros((2,240,320),bool);mask[0,1:3,1:3]=True
 row=dict(global_tick=0,elapsed_episode_s=0.,kind='query',RGB_sha256='CPU_fixture_only',raw_logits_sha256='CPU_fixture_only',masks=mask,presence=np.asarray([.9,0.]))
 class Original:
  def __init__(self):self.owner=SimpleNamespace(memory=SimpleNamespace(rows=[row]),actual_tracking_calls=1,seed=mask);self.source_query=0
  def finish(self):calls.append(1);return token
 assert tracked(Original,fixture)().finish()is token and calls==[1]
 recorded=read(fixture/'tracking_equivalence_source000.json');assert recorded['no_extra_model_forward']and recorded['actual_tracking_calls']==1
 from native_onpolicy_calibration_data_v26.reader import Reader
 original_root=mapped(read(T/'outputs/native_onpolicy_collection_v38/fit_007/execution_state.json')['root']);s,c,a=Reader().qualified(original_root);assert len(c)==25 and a['passed']
 failed=mapped(read(T/'outputs/native_onpolicy_arm_collection_v29/val_000/execution_state.json')['root'])
 try:Reader().qualified(failed)
 except (AssertionError,FileNotFoundError,ValueError):pass
 else:raise AssertionError('Failed29 scientific outputs accepted')
 state=T/'outputs/native_object_state_storage_v2/actual_report.json';sv=read(state);verify(sv);assert sv['passed']and sv['own_GPU_released'];assert read(state.with_name('CPU_GPU_host_actual_wait_v1.json'))['exit_code']==0
 camera=T/'outputs/native_used_cameras_v1/CPU_contract_checks_v1.json';assert read(camera)['passed'];verify(read(camera))
 reader=T/'native_onpolicy_calibration_data_v26';reader_out=T/'outputs/native_onpolicy_calibration_data_v26';reader_out.mkdir(parents=True,exist_ok=True)
 inputs=[proof,proof.with_name('CPU_actual_host_exit_v3.json'),state,state.with_name('CPU_GPU_host_actual_wait_v1.json'),camera,original_root/'atomic_full_recovery_qualification.json']
 rp=reader_out/'CPU_reader_regression_v1.json'
 write(rp,dict(passed=True,actual_qualified_C38_25_retained=True,failed29_refused=True,original325_distinct_support_and_old100_unchanged=True,
  no_GPU_optimizer_or_fixture_native_qualification=True,source_sha256={str(p):sha(p)for p in reader.iterdir()if p.is_file()},input_sha256={str(p):sha(p)for p in inputs}))
 sources=[p for folder in[new,Path(__file__).parent]for p in folder.rglob('*')if p.is_file()and p.suffix in('.py','.sh','.md')]
 write(out/'CPU_contract_checks_v1.json',dict(passed=True,original29_math_controller_data_and_guards_preserved=True,
  actual_startup10=startup,actual_RNG_dispatch10=routing,source_finish_once_return_identity_and_fingerprints=True,
  additional_whole25_raw_source_controls_mandatory=True,source_fixture_is_NOT_native_evidence=True,
  state_storage_actual_GPU_bridge_passed=True,used_camera_native_GPU_bridge_STILL_REQUIRED_before_study=True,
  no_GPU_model_optimizer_or_scientific_qualification=True,source_sha256={str(p):sha(p)for p in sources},input_sha256={str(p):sha(p)for p in [*inputs,rp]}))
 assert not torch.cuda.is_initialized()
 print('ARM31_TEN_LANES_RNG_AND_RAW_SOURCE_CPU_PASS_FRONT_GPU_PARITY_GATE_REMAINS',flush=True)
if __name__=='__main__':main()
