"""Only a qualified NEW full TRAIN capture may supply a recovery state."""
from pathlib import Path
import numpy as np
from .common import read,sha,mapped,T

def arrays(p):
 with np.load(mapped(p),allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def load(spec,item):
 root=mapped(spec['full_root']);fullspec=read(root/'protocol.json');terminal=read(root/'terminal_audit.json')
 if fullspec['scope']!='frozen_full_onpolicy_TRAIN_capture'or fullspec['actual_seed_start']!=7120000:raise ValueError('Only new on-policy FIT capture; no old development/EVAL relabel')
 assert terminal['passed']and terminal['batch_protocol_identity']==fullspec['identity']
 folder=mapped(item['full_folder']);assert folder.is_relative_to(root/'cases')and folder.name=='M_event_full'
 result=read(folder/'result.json');assert result['complete']and result['condition']=='M_event_full'
 index=read(folder/'offline_feedback_index.json');assert dict(packet=item['packet'],action_query=item['action_query'],application_mode=item['application_mode'],actual_teacher_label_available=False)in index['potential_actions']
 q=item['action_query'];assert type(q)is int and 0<q<len(result['queries']);chunks=[];prefix=[];count=0
 for i,row in enumerate(result['queries'][:q]):
  assert row['query']==i and row['execution']['actual_complete_native32']and row['execution']['commands_returned']==32
  assert sha(mapped(row['actions_file']))==row['actions_sha256'];a=arrays(row['actions_file'])['actions'];e=row['execution'];assert sha(mapped(e['actual_trace_file']))==e['actual_trace_sha256']
  assert np.array_equal(a,arrays(e['actual_trace_file'])['source_commands']);chunks.append(a.copy());meta=read(folder/('execution%03d.json'%i));sources=[read(f)['source_query']for f in folder.glob('source*/source.json')];selected=row['event']['decision']['selected_event'];kind=None
  if i in sources:
   kinds=selected['kinds'];kind=kinds[0]if len(kinds)==1 else tuple(kinds)
  prefix.append(dict(actions=a.copy(),metadata=meta,kind=kind,actual_trace=arrays(e['actual_trace_file'])));count+=32
 row=result['queries'][q];observation=row['observation'];policy=row['policy'];assert sha(mapped(observation['file']))==observation['sha256']and sha(mapped(policy['policy_file']))==policy['policy_sha256']
 env=read(folder/('query%03d_environment_audit.json'%q));assert env['native_environment_unchanged']and env['no_diagnostic_forward_inputs']
 return dict(chunks=chunks,prefix=prefix,query=q,expected_observation=arrays(observation['file']),expected_physical=env['physical_sha256'],expected_rng=env['rng_sha256'],expected_native_action_count=count,policy=policy,once_policy=arrays(policy['policy_file']),item=item)

def require_same(observed,diagnostic,p):
 e=p['expected_observation'];fail=[]
 if observed['native_action_count']!=p['expected_native_action_count']:fail.append('native_counter')
 if not np.array_equal(observed['command_state'],e['command_state']):fail.append('own_command')
 if len(observed['rgb'])!=3 or any(not np.array_equal(observed['rgb'][i],e['cam'+str(i)])for i in range(3)):fail.append('RGB')
 if diagnostic['physical_sha256']!=p['expected_physical']:fail.append('physics')
 if diagnostic['rng_sha256']!=p['expected_rng']:fail.append('RNG')
 if fail:raise ValueError('Same-state recovery refused: '+','.join(fail))
 return dict(passed=True,query=p['query'],native_counter=p['expected_native_action_count'],RGB_command_physics_RNG_exact=True,offline_diagnostics_not_forward_inputs=True)
