"""Saved full-policy CPU replay: ordering and commands, not VLA/native execution."""
import copy,hashlib,json,types,shutil
from pathlib import Path
import numpy as np,torch
from native_policy_execution_v1.common import T,read,write,sha
from native_onpolicy_recovery_v3.common import mapped
from native_feedback_application_v1.episode import execute as original
from native_arm_phase_feedback_v1.episode import execute as arm_original
from cx_tta.multi_event import EventConfig
from .episode import make_execute,source

def arrays(path):
 with np.load(mapped(path),allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}

def main():
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();source(original);source(arm_original)
 out=T/'outputs/native_q0_policy_overlap_v1';out.mkdir(parents=True,exist_ok=True);fixture=out/'CPU_recorded_policy_replay_v2';assert not fixture.exists();fixture.mkdir()
 stage=T/'outputs/native_onpolicy_collection_v25/fit_005';root=mapped(read(stage/'execution_state.json')['root']);assert read(stage/'actual_host_wait_exit.json')['exit_code']==0
 spec=read(root/'protocol.json');inputs={};records=[]
 selected=[]
 for task in spec['tasks']:
  cases=sorted((root/'accepted').glob(task+':*.json'),key=lambda p:read(p)['episode_index']);selected.append(cases[0])
 for accepted in selected:
  case=read(accepted);folder=root/'cases'/case['case_id']/'M_event_full';result=read(folder/'result.json');queries=result['queries'];assert queries
  def run(mode):
   target=fixture/(case['case_id']+'_'+mode);task=types.SimpleNamespace(take_action_cnt=0,step_lim=result['original_native_step_limit'],eval_success=False,q=0);calls=[];captures=[];events=[]
   def observed():
    row=queries[min(task.q,len(queries)-1)];z=arrays(row['observation']['file']);return dict(rgb=tuple(z['cam'+str(i)]for i in range(3)),command_state=z['command_state'],native_action_count=task.take_action_cnt)
   initial=observed();expected=dict(rgb=initial['rgb'],command_state=initial['command_state'],instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest(),available=False)
   class Actor:
    instruction=case['instruction']
    def query(self,current,record,destination):
     q=record['query'];assert q==len(calls)==task.q;calls.append(q);events.append(('query',q));row=queries[q]['policy'];value=arrays(row['policy_file']);policy=dict(row);policy.update(observation_file=record['file'],observation_sha256=record['sha256'],policy_file=str(destination/('policy%03d.npz'%q)))
     shutil.copyfile(mapped(row['policy_file']),policy['policy_file']);policy['policy_sha256']=sha(policy['policy_file'])
     proposal=dict(base=value['base'],raw_delta=value['raw_delta'],legal={k[6:]:v for k,v in value.items()if k.startswith('legal_')})
     return proposal,policy,value
    def finish_chunk(self,actions,metadata,tracefile,destination):
     return dict(queries[task.q-1]['execution'],actual_trace_file=str(tracefile),actual_trace_sha256=sha(tracefile))
   actor=Actor()
   def capture(task,actions,**kwargs):
    q=task.q;events.append(('capture',q));trace=arrays(queries[q]['execution']['actual_trace_file']);assert np.array_equal(actions,trace['source_commands']),('Recorded full commands changed',case['case_id'],q,mode)
    captures.append(actions.copy());metadata=read(folder/('execution%03d.json'%q));task.q+=1
    if task.q==len(queries):task.take_action_cnt=result['native_action_count'];task.eval_success=result['success']
    else:task.take_action_cnt+=queries[q]['execution']['commands_returned']
    return trace,dict(metadata),None
   def bootstrap(record,policy):
    assert calls==[0]and not captures;assert record['query']==policy['query']==0;events.append(('semantic',0));return expected,False
   namespace=dict(original.__globals__,capture=capture);eager=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__);eager.__kwdefaults__=original.__kwdefaults__
   # Keep the actual original function source and change only this CPU replay capture dependency.
   deferred=make_execute(original);deferred.__globals__['capture']=capture
   fn=deferred if mode=='deferred'else eager
   kwargs=dict(condition='M_event_full',event_config=EventConfig(**spec['event_config']),response_model=None,qualification=mapped(spec['trained_qualification']),predictor=None,source_factory=lambda *a:(_ for _ in ()).throw(AssertionError('No semantic feedback in CPU replay')),seed_available=None if mode=='deferred'else False,expected_initial=None if mode=='deferred'else expected,instruction=case['instruction'],folder=target,cpu_fixture=True)
   if mode=='deferred':kwargs.update(semantic_bootstrap=bootstrap,bootstrap_initial=initial)
   value=fn(task,actor,observed,**kwargs)
   assert value['CPU_fixture_only']and value['native_action_count']==result['native_action_count']and value['success']==result['success']and calls==list(range(len(queries)))and len(captures)==len(queries)
   if mode=='deferred':assert events[:3]==[('query',0),('semantic',0),('capture',0)]and events.count(('semantic',0))==1
   return captures,value
  a,av=run('eager');b,bv=run('deferred');assert len(a)==len(b)and all(np.array_equal(x,y)for x,y in zip(a,b))
  records.append(dict(case_id=case['case_id'],queries=len(a),all_recorded_full_commands_byte_exact=True,semantic_wait_after_once_q0_before_first_capture=True))
  for f in [accepted,folder/'result.json',*[mapped(q['policy']['policy_file'])for q in queries],*[mapped(q['execution']['actual_trace_file'])for q in queries]]:inputs[str(f)]=sha(f)
 assert not torch.cuda.is_initialized()
 write(out/'CPU_recorded_policy_order_checks_v2.json',dict(passed=True,records=records,actual_recorded_full_policy_and_trace_replay=True,actual_VLA_or_native_physics_NOT_rerun=True,mask_feedback_DISABLED_for_CPU_composition_replay=True,no_GPU_optimizer_or_native_qualification=True,source_sha256={str(p):sha(p)for p in Path(__file__).parent.glob('*')if p.is_file()},input_sha256=inputs))
 print('CPU_RECORDED_POLICY_Q0_ORDER_AND_FULL_COMMANDS_PASS',records,flush=True)

if __name__=='__main__':main()
